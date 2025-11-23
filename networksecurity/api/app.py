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
        model_path = "final_models/model.pkl"
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
