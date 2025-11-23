import os
import sys
import io
import glob
import asyncio
import logging
from base64 import b64encode
from datetime import datetime
from contextlib import asynccontextmanager, redirect_stdout, redirect_stderr
import threading

import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# --- 项目模块导入 ---
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constant import training_pipeline
from networksecurity.utils.ml_utils.data_validator import DataValidator

# --- 可视化库导入 ---
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# --- 【已补全】WebSocket 及流重定向相关类的完整定义 ---

class ConnectionManager:
    """管理所有活跃的WebSocket连接。"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        if not self.active_connections:
            return
        # 清理消息中的空行和多余空格
        cleaned_message = "\n".join(line for line in message.strip().splitlines() if line.strip())
        if cleaned_message:
            await asyncio.gather(*[conn.send_text(cleaned_message) for conn in self.active_connections],
                                 return_exceptions=True)


# 创建一个全局的连接管理器实例
manager = ConnectionManager()


class WebSocketLogHandler(logging.Handler):
    """一个将日志记录转发到WebSocket的处理器。"""

    def __init__(self, manager_instance: ConnectionManager):
        super().__init__()
        self.manager = manager_instance

    def emit(self, record):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.manager.broadcast(self.format(record)), loop)
        except RuntimeError:
            pass


class StreamToLogger:
    """一个伪文件流，其write方法会将消息路由到日志记录器。"""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# --- FastAPI 应用实例和生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 配置日志系统
    formatter = logging.Formatter("%(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    ws_handler = WebSocketLogHandler(manager)
    ws_handler.setLevel(logging.INFO)
    ws_handler.setFormatter(formatter)
    root_logger.addHandler(ws_handler)

    print("应用启动中...")
    yield
    print("应用正在关闭。")


app = FastAPI(title="网络安全威胁预测 API", version="7.1.0 (fixed)", lifespan=lifespan)

# --- 挂载静态文件和中间件 ---
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
templates = Jinja2Templates(directory="Templates")


# --- WebSocket 端点 ---
@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await manager.broadcast("--- [SYSTEM] 前端控制台连接成功，等待指令... ---")
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- 异步训练函数 ---
async def run_training_pipeline():
    # 操作系统层面的管道重定向
    r, w = os.pipe()
    stdout_original = os.dup(1)
    stderr_original = os.dup(2)
    os.dup2(w, 1)
    os.dup2(w, 2)

    stop_reader_event = threading.Event()
    loop = asyncio.get_running_loop()

    def pipe_reader():
        while not stop_reader_event.is_set():
            try:
                data = os.read(r, 1024)
                if data:
                    message = data.decode('utf-8', errors='ignore')
                    loop.call_soon_threadsafe(asyncio.create_task, manager.broadcast(message))
                else:  # 当写入端关闭时，read会返回空字节
                    break
            except Exception:
                break
        os.close(r)

    def training_task():
        try:
            TrainingPipeline().run_pipeline()
            logging.info("✅ [FINISH] 模型演化流程执行完毕！")
        except Exception as e:
            logging.error(f"❌ [ERROR] 训练管道线程内部发生严重错误: {e}", exc_info=True)
        finally:
            os.close(w)  # 关闭写入端，这将导致读取端的os.read返回空，从而结束pipe_reader循环
            stop_reader_event.set()

    reader_thread = threading.Thread(target=pipe_reader)
    reader_thread.start()

    try:
        await loop.run_in_executor(None, training_task)
    finally:
        reader_thread.join()  # 等待读取线程完全结束
        os.dup2(stdout_original, 1)
        os.dup2(stderr_original, 2)
        os.close(stdout_original)
        os.close(stderr_original)
        logging.info("--- [SYSTEM] 流重定向已恢复 ---")


# --- 页面路由 ---
@app.get("/", tags=["Frontend"], response_class=HTMLResponse)
async def serve_home(request: Request): return templates.TemplateResponse("index.html",
                                                                          {"request": request, "page": "home"})


@app.get("/training-console", tags=["Frontend"], response_class=HTMLResponse)
async def serve_training_console(request: Request): return templates.TemplateResponse("training.html",
                                                                                      {"request": request,
                                                                                       "page": "training"})


@app.get("/pipeline-explorer", tags=["Frontend"], response_class=HTMLResponse)
async def serve_pipeline_explorer(request: Request): return templates.TemplateResponse("pipeline.html",
                                                                                       {"request": request,
                                                                                        "page": "pipeline"})


@app.get("/evaluation-report", tags=["Frontend"], response_class=HTMLResponse)
async def serve_evaluation_report(request: Request): return templates.TemplateResponse("evaluation.html",
                                                                                       {"request": request,
                                                                                        "page": "evaluation"})


@app.get("/live-inference", tags=["Frontend"], response_class=HTMLResponse)
async def serve_live_inference(request: Request): return templates.TemplateResponse("predict.html", {"request": request,
                                                                                                     "page": "predict"})


# --- 新增的现代化页面路由 ---
@app.get("/predict", tags=["Frontend"], response_class=HTMLResponse)
async def serve_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "page": "predict"})


@app.get("/train", tags=["Frontend"], response_class=HTMLResponse)
async def serve_train(request: Request):
    return templates.TemplateResponse("training.html", {"request": request, "page": "training"})


@app.get("/tutorial", tags=["Frontend"], response_class=HTMLResponse)
async def serve_tutorial(request: Request):
    return templates.TemplateResponse("tutorial.html", {"request": request, "page": "tutorial"})


# --- Pydantic 模型 ---
class PredictionInput(BaseModel):
    having_IP_Address: int;
    URL_Length: int;
    Shortining_Service: int;
    having_At_Symbol: int;
    double_slash_redirecting: int;
    Prefix_Suffix: int;
    having_Sub_Domain: int;
    SSLfinal_State: int;
    Domain_registeration_length: int;
    Favicon: int;
    port: int;
    HTTPS_token: int;
    Request_URL: int;
    URL_of_Anchor: int;
    Links_in_tags: int;
    SFH: int;
    Submitting_to_email: int;
    Abnormal_URL: int;
    Redirect: int;
    on_mouseover: int;
    RightClick: int;
    popUpWidnow: int;
    Iframe: int;
    age_of_domain: int;
    DNSRecord: int;
    web_traffic: int;
    Page_Rank: int;
    Google_Index: int;
    Links_pointing_to_page: int;
    Statistical_report: int


# --- 核心API端点 ---
@app.post("/api/train", tags=["Training"])
async def trigger_training():
    """触发模型训练管道"""
    asyncio.create_task(run_training_pipeline())
    return {"status": "success", "message": "模型训练任务已在后台启动"}


# --- 数据上传和验证API ---
@app.get("/api/features/requirements", tags=["Data"])
async def get_feature_requirements():
    """获取训练数据特征要求"""
    validator = DataValidator()
    return validator.get_feature_requirements()


@app.post("/api/data/validate", tags=["Data"])
async def validate_data(file: UploadFile = File(...)):
    """验证上传的数据文件"""
    try:
        # 读取上传的CSV文件
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 验证数据
        validator = DataValidator()
        is_valid, report = validator.validate_features(df)

        # 获取补全建议
        imputation_suggestions = validator.suggest_imputation_strategy(df)

        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "is_valid": is_valid,
            "validation_report": report,
            "imputation_suggestions": imputation_suggestions
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"数据验证失败: {str(e)}"}
        )


@app.post("/api/data/impute", tags=["Data"])
async def impute_data(
    file: UploadFile = File(...),
    strategy: str = Form("constant"),
    fill_value: int = Form(0)
):
    """补全数据特征"""
    try:
        # 读取上传的CSV文件
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 补全数据
        validator = DataValidator()
        df_imputed, impute_report = validator.impute_missing_features(
            df,
            strategy=strategy,
            fill_value=fill_value
        )

        # 保存补全后的数据
        output_path = f"uploads/imputed_{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        df_imputed.to_csv(output_path, index=False)

        return {
            "status": "success",
            "message": "数据补全成功",
            "output_file": output_path,
            "impute_report": impute_report,
            "rows": len(df_imputed),
            "columns": len(df_imputed.columns)
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"数据补全失败: {str(e)}"}
        )


@app.get("/api/data/download/{filename}", tags=["Data"])
async def download_imputed_data(filename: str):
    """下载补全后的数据"""
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/csv'
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "文件不存在"}
        )


@app.get("/predict_on_test_data", tags=["Prediction"])
async def predict_on_test_data():
    try:
        artifact_dir = training_pipeline.ARTIFACT_DIR;
        list_of_runs = glob.glob(os.path.join(artifact_dir, "*"))
        if not list_of_runs: raise FileNotFoundError("未找到任何训练产物目录。")
        latest_run_dir = max(list_of_runs, key=os.path.getctime)
        test_file_path = os.path.join(latest_run_dir, "data_ingestion", "ingested", "test.csv")
        if not os.path.exists(test_file_path): raise FileNotFoundError(
            f"在最新的训练产物中未找到 test.csv: {test_file_path}")
        df = pd.read_csv(test_file_path)
        model_path = os.path.join("final_models", "model.pkl");
        preprocessor_path = os.path.join("final_models", "preprocessor.pkl")
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path): raise FileNotFoundError(
            "模型或预处理器未在 'final_models' 目录中找到。请先成功运行一次训练管道。")
        model = load_object(file_path=model_path);
        preprocessor = load_object(file_path=preprocessor_path)
        y_true = df[training_pipeline.TARGET_COLUMN].replace(-1, 0)
        df_to_predict = df.drop(columns=[training_pipeline.TARGET_COLUMN], axis=1, errors='ignore')
        transformed_df = preprocessor.transform(df_to_predict);
        y_pred = model.predict(transformed_df)
        df['prediction'] = np.where(y_pred == 1, '危险 (Malicious)', '安全 (Benign)')
        table_data = df.to_dict(orient='records');
        plt.style.use('dark_background')
        cm = confusion_matrix(y_true, y_pred);
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Malicious (1)'])
        fig, ax = plt.subplots(figsize=(6, 5));
        disp.plot(ax=ax, cmap='magma', colorbar=False);
        ax.set_title("混淆矩阵 (Confusion Matrix)")
        buf_cm = io.BytesIO();
        plt.savefig(buf_cm, format='png', bbox_inches='tight', transparent=True);
        plt.close(fig)
        img_cm_b64 = b64encode(buf_cm.getvalue()).decode('utf-8');
        pred_counts = df['prediction'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 5));
        ax.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90, colors=['#39d353', '#f85149'])
        ax.set_title("测试集预测结果分布");
        ax.axis('equal')
        buf_pie = io.BytesIO();
        plt.savefig(buf_pie, format='png', bbox_inches='tight', transparent=True);
        plt.close(fig)
        img_pie_b64 = b64encode(buf_pie.getvalue()).decode('utf-8')
        return JSONResponse(
            content={"table_data": table_data, "img_confusion_matrix": f"data:image/png;base64,{img_cm_b64}",
                     "img_pie_chart": f"data:image/png;base64,{img_pie_b64}"})
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


@app.post("/predict_live", tags=["Prediction"])
async def predict_live(input_data: PredictionInput):
    try:
        model_path = os.path.join("final_models", "model.pkl");
        preprocessor_path = os.path.join("final_models", "preprocessor.pkl")
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path): raise FileNotFoundError(
            "模型或预处理器未在 'final_models' 目录中找到。")
        model = load_object(file_path=model_path);
        preprocessor = load_object(file_path=preprocessor_path)
        input_df = pd.DataFrame([input_data.dict()])
        transformed_df = preprocessor.transform(input_df);
        prediction_result = model.predict(transformed_df)
        result_label = "危险 (Malicious)" if prediction_result[0] == 1 else "安全 (Benign)"
        return JSONResponse(content={"prediction": result_label, "raw_prediction": int(prediction_result[0])})
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --- 异常处理器和启动命令 ---
@app.exception_handler(NetworkSecurityException)
async def network_security_exception_handler(request: Request, exc: NetworkSecurityException): return JSONResponse(
    status_code=500, content={"message": str(exc)})


if __name__ == "__main__":
    uvicorn.run("test_app:app", host="0.0.0.0", port=8000, reload=True)