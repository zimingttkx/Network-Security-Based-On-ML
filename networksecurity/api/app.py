"""
优化的FastAPI应用
包含：异步处理、API版本化、限流、监控等企业级功能
"""
import os
import sys
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import pandas as pd
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.config.config_manager import get_config_manager
from networksecurity.utils.ml_utils.model_explanation import ModelExplainer
from networksecurity.utils.url_feature_extractor import URLFeatureExtractor


# ==================== Prometheus Metrics ====================
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['status']
)

TRAINING_COUNT = Counter(
    'training_jobs_total',
    'Total training jobs',
    ['status']
)


# ==================== Request/Response Models ====================
class PredictionRequest(BaseModel):
    """预测请求模型"""
    data: List[List[float]] = Field(..., description="特征数据")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0, 4.0]]
            }
        }


class PredictionResponse(BaseModel):
    """预测响应模型"""
    predictions: List[int] = Field(..., description="预测结果")
    probabilities: Optional[List[float]] = Field(None, description="预测概率")
    threat_level: List[str] = Field(..., description="威胁级别")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    uptime: float


class TrainingResponse(BaseModel):
    """训练响应"""
    status: str
    message: str
    metrics: Optional[dict] = None


class FeatureContribution(BaseModel):
    """特征贡献模型"""
    feature_name: str = Field(..., description="特征名称")
    shap_value: float = Field(..., description="SHAP值")
    contribution: str = Field(..., description="贡献方向 (positive/negative)")


class ExplainResponse(BaseModel):
    """解释响应模型"""
    predictions: List[int] = Field(..., description="预测结果")
    probabilities: Optional[List[float]] = Field(None, description="预测概率")
    threat_level: List[str] = Field(..., description="威胁级别")
    explanations: List[dict] = Field(..., description="每个样本的特征贡献分析")


class URLRequest(BaseModel):
    """URL预测请求模型"""
    url: str = Field(..., description="要检测的URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com"
            }
        }


class URLPredictionResponse(BaseModel):
    """URL预测响应模型"""
    url: str = Field(..., description="检测的URL")
    prediction: int = Field(..., description="预测结果 (0: 安全, 1: 危险)")
    probability: Optional[float] = Field(None, description="危险概率")
    threat_level: str = Field(..., description="威胁级别")
    features: dict = Field(..., description="提取的特征")
    feature_extraction_time: float = Field(..., description="特征提取耗时(秒)")


# ==================== Lifespan Context Manager ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logging.info("应用启动中...")
    app.state.start_time = time.time()
    app.state.config = get_config_manager()

    # 加载模型（如果存在）
    try:
        model_path = "models/model.pkl"
        if os.path.exists(model_path):
            app.state.model = load_object(model_path)
            logging.info("模型加载成功")
        else:
            app.state.model = None
            logging.warning("模型文件不存在，请先训练模型")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        app.state.model = None

    yield

    # Shutdown
    logging.info("应用关闭中...")


# ==================== FastAPI Application ====================
config_manager = get_config_manager()
api_config = config_manager.get('api', {})

app = FastAPI(
    title=api_config.get('title', 'Network Security API'),
    description=api_config.get('description', 'Network Threat Detection API'),
    version=api_config.get('version', '2.0.0'),
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# ==================== Middleware ====================
# CORS
cors_config = api_config.get('cors', {})
if cors_config.get('enabled', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get('allow_origins', ["*"]),
        allow_credentials=cors_config.get('allow_credentials', True),
        allow_methods=cors_config.get('allow_methods', ["*"]),
        allow_headers=cors_config.get('allow_headers', ["*"]),
    )

# Gzip压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ==================== Request Tracking Middleware ====================
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """追踪请求指标"""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    response.headers["X-Process-Time"] = str(duration)
    return response


# ==================== Health Check ====================
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """健康检查端点"""
    uptime = time.time() - request.app.state.start_time
    return HealthResponse(
        status="healthy",
        version=app.version,
        uptime=uptime
    )


@app.get("/ready", tags=["System"])
async def readiness_check(request: Request):
    """就绪检查"""
    if request.app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未加载"
        )
    return {"status": "ready"}


# ==================== Metrics Endpoint ====================
@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus指标端点"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ==================== Training Endpoint ====================
@app.post("/api/v1/train", response_model=TrainingResponse, tags=["Training"])
async def train_model():
    """启动模型训练"""
    try:
        logging.info("收到训练请求")
        TRAINING_COUNT.labels(status='started').inc()

        # 异步执行训练（生产环境应使用Celery等任务队列）
        training_pipeline = TrainingPipeline()
        artifact = training_pipeline.run_pipeline()

        TRAINING_COUNT.labels(status='success').inc()

        return TrainingResponse(
            status="success",
            message="训练完成",
            metrics={
                "train_f1": float(artifact.train_metric_artifact.f1_score),
                "test_f1": float(artifact.test_metric_artifact.f1_score)
            }
        )

    except Exception as e:
        TRAINING_COUNT.labels(status='failed').inc()
        logging.error(f"训练失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"训练失败: {str(e)}"
        )


# ==================== Prediction Endpoints ====================
@app.post("/api/v1/predict/file", tags=["Prediction"])
async def predict_from_file(file: UploadFile = File(...)):
    """从CSV文件预测"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="仅支持CSV文件"
            )

        # 读取文件
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # 检查模型
        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未加载，请先训练模型"
            )

        # 预测
        predictions = app.state.model.predict(df)
        df['prediction'] = predictions
        df['threat_level'] = np.where(predictions == 1, '危险 (Malicious)', '安全 (Benign)')

        PREDICTION_COUNT.labels(status='success').inc()

        return JSONResponse(content=df.to_dict(orient='records'))

    except Exception as e:
        PREDICTION_COUNT.labels(status='failed').inc()
        logging.error(f"预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """从JSON数据预测"""
    try:
        # 检查模型
        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未加载，请先训练模型"
            )

        # 转换数据
        df = pd.DataFrame(request.data)

        # 预测
        predictions = app.state.model.predict(df)
        threat_levels = ['危险 (Malicious)' if p == 1 else '安全 (Benign)' for p in predictions]

        # 获取概率（如果模型支持）
        probabilities = None
        if hasattr(app.state.model, 'predict_proba'):
            proba = app.state.model.predict_proba(df)
            probabilities = proba[:, 1].tolist()

        PREDICTION_COUNT.labels(status='success').inc()

        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            threat_level=threat_levels
        )

    except Exception as e:
        PREDICTION_COUNT.labels(status='failed').inc()
        logging.error(f"预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )


# ==================== Explanation Endpoint ====================
@app.post("/api/v1/explain", response_model=ExplainResponse, tags=["Explanation"])
async def explain_prediction(request: PredictionRequest):
    """预测并解释结果 - 返回预测结果及Top 5特征贡献分析"""
    try:
        # 检查模型
        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未加载，请先训练模型"
            )

        # 转换数据
        df = pd.DataFrame(request.data)
        
        # 预测
        predictions = app.state.model.predict(df)
        threat_levels = ['危险 (Malicious)' if p == 1 else '安全 (Benign)' for p in predictions]
        
        # 获取概率
        probabilities = None
        if hasattr(app.state.model, 'predict_proba'):
            proba = app.state.model.predict_proba(df)
            probabilities = proba[:, 1].tolist()
        
        # 生成SHAP解释
        explainer = ModelExplainer()
        explanation_result = explainer.get_explanation(app.state.model, df.values)
        
        # 提取每个样本的Top 5特征贡献
        sample_explanations = []
        if explanation_result["success"]:
            for exp in explanation_result["explanations"]:
                top_5_features = exp["top_features"][:5]
                sample_explanations.append({
                    "sample_index": exp["sample_index"],
                    "base_value": exp["base_value"],
                    "top_5_contributions": [
                        {
                            "feature_name": f["name"],
                            "feature_value": f["value"],
                            "shap_value": f["shap_value"],
                            "contribution": f["contribution"]
                        }
                        for f in top_5_features
                    ]
                })
        else:
            logging.warning(f"SHAP解释生成失败: {explanation_result.get('error', 'Unknown error')}")
            sample_explanations = [{"error": "无法生成特征解释"}]
        
        PREDICTION_COUNT.labels(status='success').inc()
        
        return ExplainResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            threat_level=threat_levels,
            explanations=sample_explanations
        )

    except Exception as e:
        PREDICTION_COUNT.labels(status='failed').inc()
        logging.error(f"预测解释失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测解释失败: {str(e)}"
        )


# ==================== URL Prediction Endpoint ====================
@app.post("/api/v1/predict/url", response_model=URLPredictionResponse, tags=["Prediction"])
async def predict_from_url(request: URLRequest):
    """
    从URL自动提取特征并预测
    
    输入一个URL，系统自动提取30个特征并返回预测结果
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    try:
        # 检查模型
        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="模型未加载，请先训练模型"
            )
        
        # 在线程池中执行特征提取（避免阻塞事件循环）
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            extractor = URLFeatureExtractor(timeout=15)
            features = await loop.run_in_executor(
                executor, 
                extractor.extract_features, 
                request.url
            )
        
        extraction_time = time.time() - start_time
        
        # 转换为DataFrame（按特征顺序）
        feature_order = [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report'
        ]
        df = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        
        # 预测
        prediction = app.state.model.predict(df)[0]
        threat_level = '危险 (Malicious)' if prediction == 1 else '安全 (Benign)'
        
        # 获取概率
        probability = None
        if hasattr(app.state.model, 'predict_proba'):
            proba = app.state.model.predict_proba(df)
            probability = float(proba[0, 1])
        
        PREDICTION_COUNT.labels(status='success').inc()
        
        return URLPredictionResponse(
            url=request.url,
            prediction=int(prediction),
            probability=probability,
            threat_level=threat_level,
            features=features,
            feature_extraction_time=round(extraction_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_COUNT.labels(status='failed').inc()
        logging.error(f"URL预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL预测失败: {str(e)}"
        )


# ==================== Exception Handlers ====================
@app.exception_handler(NetworkSecurityException)
async def network_security_exception_handler(request: Request, exc: NetworkSecurityException):
    """自定义异常处理器"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logging.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "内部服务器错误"}
    )


# ==================== Main ====================
if __name__ == "__main__":
    app_config = config_manager.get('app', {})

    uvicorn.run(
        "networksecurity.api.app:app",
        host=app_config.get('host', '0.0.0.0'),
        port=app_config.get('port', 8000),
        reload=app_config.get('debug', False),
        workers=1
    )
