# 网络安全威胁检测系统 - API 完整文档

> **版本**: v7.1.0  
> **最后更新**: 2024年12月  
> **维护者**: Network Security Team

---

## 目录

1. [快速入门](#快速入门)
2. [API概览](#api概览)
3. [认证与安全](#认证与安全)
4. [核心API端点](#核心api端点)
   - [威胁预测API](#威胁预测api)
   - [模型训练API](#模型训练api)
   - [数据管理API](#数据管理api)
5. [WebSocket实时通信](#websocket实时通信)
6. [错误处理](#错误处理)
7. [SDK与代码示例](#sdk与代码示例)
8. [最佳实践](#最佳实践)
9. [FAQ常见问题](#faq常见问题)

---

## 快速入门

### 30秒快速体验

```bash
# 1. 启动服务
python test_app.py

# 2. 测试API是否正常
curl http://localhost:8000/

# 3. 进行一次威胁预测
curl -X POST http://localhost:8000/predict_live \
  -H "Content-Type: application/json" \
  -d '{"having_IP_Address":1,"URL_Length":1,"Shortining_Service":1,"having_At_Symbol":1,"double_slash_redirecting":1,"Prefix_Suffix":-1,"having_Sub_Domain":1,"SSLfinal_State":1,"Domain_registeration_length":1,"Favicon":1,"port":1,"HTTPS_token":1,"Request_URL":1,"URL_of_Anchor":1,"Links_in_tags":1,"SFH":1,"Submitting_to_email":1,"Abnormal_URL":1,"Redirect":0,"on_mouseover":1,"RightClick":1,"popUpWidnow":1,"Iframe":1,"age_of_domain":1,"DNSRecord":1,"web_traffic":1,"Page_Rank":1,"Google_Index":1,"Links_pointing_to_page":1,"Statistical_report":1}'
```

### 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Python | 3.10 | 3.12+ |
| FastAPI | 0.100.0 | 最新版 |
| Uvicorn | 0.22.0 | 最新版 |

### 服务地址

| 环境 | 地址 | 说明 |
|------|------|------|
| 本地开发 | `http://localhost:8000` | 默认开发环境 |
| 本地开发(备用) | `http://127.0.0.1:8000` | 如localhost无法访问时使用 |

---

## API概览

### 基础信息

```
Base URL: http://localhost:8000
Content-Type: application/json
API Version: v7.1.0
```

### 端点总览

| 方法 | 端点 | 功能 | 标签 |
|------|------|------|------|
| `POST` | `/predict_live` | 实时威胁预测 | Prediction |
| `GET` | `/predict_on_test_data` | 测试数据批量预测 | Prediction |
| `POST` | `/api/train` | 触发模型训练 | Training |
| `GET` | `/api/features/requirements` | 获取特征要求 | Data |
| `POST` | `/api/data/validate` | 验证数据文件 | Data |
| `POST` | `/api/data/impute` | 补全缺失特征 | Data |
| `GET` | `/api/data/download/{filename}` | 下载处理后的数据 | Data |
| `WS` | `/ws/train` | 训练日志实时推送 | WebSocket |

### 页面路由

| 路径 | 功能 | 说明 |
|------|------|------|
| `/` | 首页 | 项目介绍和导航 |
| `/predict` | 威胁预测页面 | 可视化预测界面 |
| `/train` | 模型训练页面 | 训练控制台 |
| `/tutorial` | 使用教程 | 新手引导 |
| `/training-console` | 训练控制台(旧版) | 兼容旧版本 |
| `/pipeline-explorer` | 管道浏览器 | 查看训练管道 |
| `/evaluation-report` | 评估报告 | 模型评估结果 |
| `/live-inference` | 实时推理(旧版) | 兼容旧版本 |

---

## 认证与安全

### 当前版本

当前版本为**开发/学习版本**，暂未启用认证机制。适用于：
- 本地开发测试
- 学习和教学环境
- 内网部署

### CORS配置

```python
# 当前配置允许所有来源（仅限开发环境）
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

> ⚠️ **生产环境警告**: 部署到生产环境前，请务必配置适当的CORS策略和认证机制。

---

## 核心API端点

### 威胁预测API

#### POST /predict_live

**功能描述**: 对单条网络流量数据进行实时威胁预测，判断是否存在安全威胁。

**使用场景**:
- 实时网络流量监控
- 用户提交的URL安全检测
- 集成到安全网关进行实时过滤

**请求格式**:

```http
POST /predict_live HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

**请求体 (Request Body)**:

```json
{
  "having_IP_Address": 1,
  "URL_Length": 1,
  "Shortining_Service": 1,
  "having_At_Symbol": 1,
  "double_slash_redirecting": 1,
  "Prefix_Suffix": -1,
  "having_Sub_Domain": 1,
  "SSLfinal_State": 1,
  "Domain_registeration_length": 1,
  "Favicon": 1,
  "port": 1,
  "HTTPS_token": 1,
  "Request_URL": 1,
  "URL_of_Anchor": 1,
  "Links_in_tags": 1,
  "SFH": 1,
  "Submitting_to_email": 1,
  "Abnormal_URL": 1,
  "Redirect": 0,
  "on_mouseover": 1,
  "RightClick": 1,
  "popUpWidnow": 1,
  "Iframe": 1,
  "age_of_domain": 1,
  "DNSRecord": 1,
  "web_traffic": 1,
  "Page_Rank": 1,
  "Google_Index": 1,
  "Links_pointing_to_page": 1,
  "Statistical_report": 1
}
```

**请求参数详解**:

| 参数名 | 类型 | 必填 | 取值范围 | 说明 |
|--------|------|------|----------|------|
| `having_IP_Address` | int | ✅ | -1, 1 | URL中是否包含IP地址。-1: 包含(可疑), 1: 不包含(正常) |
| `URL_Length` | int | ✅ | -1, 0, 1 | URL长度。1: 正常(<54), 0: 可疑(54-75), -1: 异常(>75) |
| `Shortining_Service` | int | ✅ | -1, 1 | 是否使用短链服务(如bit.ly)。-1: 是(可疑), 1: 否 |
| `having_At_Symbol` | int | ✅ | -1, 1 | URL中是否包含@符号。-1: 包含(可疑), 1: 不包含 |
| `double_slash_redirecting` | int | ✅ | -1, 1 | 是否有双斜杠重定向。-1: 是(可疑), 1: 否 |
| `Prefix_Suffix` | int | ✅ | -1, 1 | 域名中是否有连字符(-)。-1: 有(可疑), 1: 无 |
| `having_Sub_Domain` | int | ✅ | -1, 0, 1 | 子域名数量。1: ≤2个, 0: 3个, -1: >3个(可疑) |
| `SSLfinal_State` | int | ✅ | -1, 0, 1 | SSL证书状态。1: 有效且可信, 0: 可疑, -1: 无效 |
| `Domain_registeration_length` | int | ✅ | -1, 1 | 域名注册时长。1: ≥1年, -1: <1年(可疑) |
| `Favicon` | int | ✅ | -1, 1 | Favicon是否从外部加载。1: 本域, -1: 外部(可疑) |
| `port` | int | ✅ | -1, 1 | 是否使用标准端口。1: 标准(80,443等), -1: 非标准 |
| `HTTPS_token` | int | ✅ | -1, 1 | 域名中是否包含HTTPS字样。-1: 包含(可疑), 1: 不包含 |
| `Request_URL` | int | ✅ | -1, 0, 1 | 外部资源请求比例。1: <22%, 0: 22-61%, -1: >61% |
| `URL_of_Anchor` | int | ✅ | -1, 0, 1 | 锚点指向外部比例。1: <31%, 0: 31-67%, -1: >67% |
| `Links_in_tags` | int | ✅ | -1, 0, 1 | Meta/Script/Link标签外部比例。1: <17%, 0: 17-81%, -1: >81% |
| `SFH` | int | ✅ | -1, 0, 1 | 表单提交地址。1: 本域, 0: 空或about:blank, -1: 外部 |
| `Submitting_to_email` | int | ✅ | -1, 1 | 是否使用mail()或mailto:。-1: 是(可疑), 1: 否 |
| `Abnormal_URL` | int | ✅ | -1, 1 | URL是否包含主机名。-1: 不包含(异常), 1: 包含 |
| `Redirect` | int | ✅ | -1, 0, 1 | 重定向次数。0: ≤1次, 1: 2-3次, -1: ≥4次(可疑) |
| `on_mouseover` | int | ✅ | -1, 1 | 是否有状态栏修改脚本。-1: 有(可疑), 1: 无 |
| `RightClick` | int | ✅ | -1, 1 | 是否禁用右键。-1: 禁用(可疑), 1: 正常 |
| `popUpWidnow` | int | ✅ | -1, 1 | 是否有带输入框的弹窗。-1: 有(可疑), 1: 无 |
| `Iframe` | int | ✅ | -1, 1 | 是否使用不可见iframe。-1: 有(可疑), 1: 无 |
| `age_of_domain` | int | ✅ | -1, 1 | 域名年龄。1: ≥6个月, -1: <6个月(可疑) |
| `DNSRecord` | int | ✅ | -1, 1 | 是否有DNS记录。1: 有, -1: 无(可疑) |
| `web_traffic` | int | ✅ | -1, 0, 1 | Alexa排名。1: <100000, 0: >100000, -1: 无排名 |
| `Page_Rank` | int | ✅ | -1, 1 | Google PageRank。1: >0.2, -1: <0.2 |
| `Google_Index` | int | ✅ | -1, 1 | 是否被Google索引。1: 已索引, -1: 未索引(可疑) |
| `Links_pointing_to_page` | int | ✅ | -1, 0, 1 | 外部链接数。1: >2, 0: 1-2, -1: 0(可疑) |
| `Statistical_report` | int | ✅ | -1, 1 | 是否在统计报告黑名单中。-1: 在(危险), 1: 不在 |

**成功响应** (HTTP 200):

```json
{
  "prediction": "安全 (Benign)",
  "raw_prediction": 0
}
```

或

```json
{
  "prediction": "危险 (Malicious)",
  "raw_prediction": 1
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `prediction` | string | 人类可读的预测结果 |
| `raw_prediction` | int | 原始预测值。0: 安全, 1: 危险 |

**错误响应**:

| HTTP状态码 | 错误类型 | 说明 |
|------------|----------|------|
| 422 | Validation Error | 请求参数格式错误或缺少必填字段 |
| 500 | Internal Server Error | 模型文件缺失或服务器内部错误 |

**422错误示例**:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "having_IP_Address"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

**500错误示例**:

```json
{
  "message": "模型或预处理器未在 'final_models' 目录中找到。"
}
```

---

#### GET /predict_on_test_data

**功能描述**: 使用最新训练产物中的测试数据集进行批量预测，并生成可视化报告（混淆矩阵、预测分布饼图）。

**使用场景**:
- 模型训练后的效果验证
- 生成模型评估报告
- 批量数据预测演示

**请求格式**:

```http
GET /predict_on_test_data HTTP/1.1
Host: localhost:8000
```

**成功响应** (HTTP 200):

```json
{
  "table_data": [
    {
      "having_IP_Address": 1,
      "URL_Length": 1,
      "Result": 0,
      "prediction": "安全 (Benign)"
    }
  ],
  "img_confusion_matrix": "data:image/png;base64,iVBORw0KGgo...",
  "img_pie_chart": "data:image/png;base64,iVBORw0KGgo..."
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `table_data` | array | 测试数据及预测结果的完整列表 |
| `img_confusion_matrix` | string | Base64编码的混淆矩阵图片 |
| `img_pie_chart` | string | Base64编码的预测分布饼图 |

---

### 模型训练API

#### POST /api/train

**功能描述**: 触发后台模型训练任务。支持传统机器学习和深度学习两种模式。

**请求格式**:

```http
POST /api/train HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

**请求体 (可选)**:

```json
{
  "use_deep_learning": false,
  "dl_model_type": "dnn",
  "dl_config": null
}
```

**请求参数详解**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `use_deep_learning` | bool | ❌ | false | 是否使用深度学习模型 |
| `dl_model_type` | string | ❌ | "dnn" | 深度学习模型类型: dnn/cnn/lstm |
| `dl_config` | object | ❌ | null | 深度学习超参数配置 |

**深度学习配置示例**:

```json
{
  "use_deep_learning": true,
  "dl_model_type": "dnn",
  "dl_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_layers": [128, 64, 32]
  }
}
```

**成功响应** (HTTP 200):

```json
{
  "status": "success",
  "message": "机器学习模型训练任务已在后台启动"
}
```

> ⚠️ **注意**: 训练任务在后台异步执行，此API立即返回。使用WebSocket `/ws/train` 获取实时训练日志。

---

### 数据管理API

#### GET /api/features/requirements

**功能描述**: 获取模型所需的全部30个特征的详细说明。

**请求格式**:

```http
GET /api/features/requirements HTTP/1.1
Host: localhost:8000
```

**成功响应** (HTTP 200):

```json
{
  "total_features": 30,
  "features": [
    {
      "name": "having_IP_Address",
      "description": "URL中是否包含IP地址 (-1: 是, 1: 否)",
      "type": "integer",
      "typical_values": "-1, 0, 1"
    }
  ]
}
```

---

#### POST /api/data/validate

**功能描述**: 验证上传的CSV数据文件，检查特征完整性、数据类型、缺失值等。

**请求格式**:

```http
POST /api/data/validate HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data
```

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `file` | File | ✅ | CSV格式的数据文件 |

**cURL示例**:

```bash
curl -X POST http://localhost:8000/api/data/validate \
  -F "file=@your_data.csv"
```

**成功响应** (HTTP 200):

```json
{
  "status": "success",
  "filename": "network_data.csv",
  "rows": 1000,
  "columns": 25,
  "is_valid": false,
  "validation_report": {
    "missing_features": ["age_of_domain", "DNSRecord"],
    "recommendations": [
      {
        "issue": "缺少 2 个特征",
        "solution": "请添加缺失特征或使用特征补全功能"
      }
    ]
  },
  "imputation_suggestions": {
    "missing_percentage": 2.5,
    "suggestions": [
      {"strategy": "mean", "reason": "缺失值较少，使用均值补全即可", "priority": 1}
    ]
  }
}
```

---

#### POST /api/data/impute

**功能描述**: 对上传的数据进行特征补全，支持多种补全策略。

**请求格式**:

```http
POST /api/data/impute HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data
```

**请求参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file` | File | ✅ | - | CSV格式的数据文件 |
| `strategy` | string | ❌ | "constant" | 补全策略: constant/mean/median/most_frequent/knn |
| `fill_value` | int | ❌ | 0 | constant策略的填充值 |

**补全策略说明**:

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `constant` | 使用固定值填充 | 缺失值较多，需要保守处理 |
| `mean` | 使用均值填充 | 缺失值较少，数据分布均匀 |
| `median` | 使用中位数填充 | 数据有异常值 |
| `most_frequent` | 使用众数填充 | 分类特征 |
| `knn` | K近邻算法填充 | 缺失值中等，需要更准确的估计 |

**成功响应** (HTTP 200):

```json
{
  "status": "success",
  "message": "数据补全成功",
  "output_file": "uploads/imputed_network_data.csv",
  "impute_report": {
    "added_features": ["age_of_domain", "DNSRecord"],
    "strategy": "knn"
  },
  "rows": 1000,
  "columns": 30
}
```

---

#### GET /api/data/download/{filename}

**功能描述**: 下载补全处理后的数据文件。

**请求格式**:

```http
GET /api/data/download/imputed_network_data.csv HTTP/1.1
Host: localhost:8000
```

**成功响应**: 返回CSV文件下载

**错误响应** (HTTP 404):

```json
{
  "status": "error",
  "message": "文件不存在"
}
```

---

## WebSocket实时通信

### WS /ws/train

**功能描述**: 建立WebSocket连接，实时接收模型训练过程中的日志输出。

**连接地址**:

```
ws://localhost:8000/ws/train
```

**使用流程**:

1. 建立WebSocket连接
2. 收到连接成功消息
3. 调用 `POST /api/train` 触发训练
4. 实时接收训练日志
5. 训练完成后收到完成消息

**JavaScript示例**:

```javascript
// 建立WebSocket连接
const ws = new WebSocket('ws://localhost:8000/ws/train');

ws.onopen = function() {
    console.log('WebSocket连接已建立');
};

ws.onmessage = function(event) {
    console.log('训练日志:', event.data);
    // 将日志显示在页面上
    document.getElementById('log-container').innerHTML += event.data + '\n';
};

ws.onerror = function(error) {
    console.error('WebSocket错误:', error);
};

ws.onclose = function() {
    console.log('WebSocket连接已关闭');
};

// 触发训练（通过HTTP API）
fetch('/api/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({use_deep_learning: false})
});
```

**Python示例**:

```python
import asyncio
import websockets
import requests

async def listen_training_logs():
    uri = "ws://localhost:8000/ws/train"
    async with websockets.connect(uri) as websocket:
        # 触发训练
        requests.post("http://localhost:8000/api/train")
        
        # 接收日志
        while True:
            try:
                message = await websocket.recv()
                print(f"[LOG] {message}")
                if "FINISH" in message:
                    break
            except websockets.exceptions.ConnectionClosed:
                break

asyncio.run(listen_training_logs())
```

**消息格式**:

| 消息类型 | 示例 | 说明 |
|----------|------|------|
| 系统消息 | `--- [SYSTEM] 前端控制台连接成功 ---` | 连接状态通知 |
| 训练日志 | `Epoch 1/100, Loss: 0.5432` | 训练过程输出 |
| 完成消息 | `✅ [FINISH] 模型演化流程执行完毕！` | 训练完成标志 |
| 错误消息 | `❌ [ERROR] 训练管道发生错误` | 错误通知 |

---

## 错误处理

### HTTP状态码说明

| 状态码 | 含义 | 处理建议 |
|--------|------|----------|
| 200 | 成功 | 正常处理响应数据 |
| 400 | 请求错误 | 检查请求参数格式 |
| 404 | 资源不存在 | 检查URL路径或资源ID |
| 422 | 验证失败 | 检查请求体字段是否完整 |
| 500 | 服务器错误 | 查看服务器日志，联系管理员 |

### 错误响应格式

**FastAPI验证错误 (422)**:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "field_name"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

**业务错误 (400/500)**:

```json
{
  "status": "error",
  "message": "具体错误描述"
}
```

**NetworkSecurityException (500)**:

```json
{
  "message": "详细的异常信息和堆栈跟踪"
}
```

### 常见错误及解决方案

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `模型或预处理器未找到` | 未训练模型 | 先执行 `/api/train` 训练模型 |
| `未找到训练产物目录` | 无历史训练记录 | 执行一次完整训练 |
| `无法解析CSV文件` | 文件格式错误 | 确保上传标准CSV格式 |
| `特征数量不匹配` | 缺少必要特征 | 使用 `/api/data/impute` 补全 |

---

## SDK与代码示例

### Python SDK封装

```python
"""
网络安全威胁检测 API 客户端
"""
import requests
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class PredictionResult:
    prediction: str
    raw_prediction: int
    is_threat: bool

class NetworkSecurityClient:
    """网络安全API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def predict(self, features: Dict[str, int]) -> PredictionResult:
        """
        进行威胁预测
        
        Args:
            features: 包含30个特征的字典
            
        Returns:
            PredictionResult: 预测结果
        """
        response = self.session.post(
            f"{self.base_url}/predict_live",
            json=features
        )
        response.raise_for_status()
        data = response.json()
        return PredictionResult(
            prediction=data['prediction'],
            raw_prediction=data['raw_prediction'],
            is_threat=data['raw_prediction'] == 1
        )
    
    def train(self, use_deep_learning: bool = False, 
              dl_model_type: str = 'dnn') -> Dict:
        """触发模型训练"""
        response = self.session.post(
            f"{self.base_url}/api/train",
            json={
                "use_deep_learning": use_deep_learning,
                "dl_model_type": dl_model_type
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_feature_requirements(self) -> Dict:
        """获取特征要求"""
        response = self.session.get(
            f"{self.base_url}/api/features/requirements"
        )
        response.raise_for_status()
        return response.json()
    
    def validate_data(self, file_path: str) -> Dict:
        """验证数据文件"""
        with open(file_path, 'rb') as f:
            response = self.session.post(
                f"{self.base_url}/api/data/validate",
                files={'file': f}
            )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """检查服务是否正常"""
        try:
            response = self.session.get(self.base_url)
            return response.status_code == 200
        except:
            return False

# 使用示例
if __name__ == "__main__":
    client = NetworkSecurityClient()
    
    # 检查服务状态
    if client.health_check():
        print("服务正常运行")
    
    # 获取特征要求
    requirements = client.get_feature_requirements()
    print(f"需要 {requirements['total_features']} 个特征")
    
    # 进行预测
    sample_features = {
        "having_IP_Address": 1, "URL_Length": 1, "Shortining_Service": 1,
        "having_At_Symbol": 1, "double_slash_redirecting": 1, "Prefix_Suffix": -1,
        "having_Sub_Domain": 1, "SSLfinal_State": 1, "Domain_registeration_length": 1,
        "Favicon": 1, "port": 1, "HTTPS_token": 1, "Request_URL": 1,
        "URL_of_Anchor": 1, "Links_in_tags": 1, "SFH": 1, "Submitting_to_email": 1,
        "Abnormal_URL": 1, "Redirect": 0, "on_mouseover": 1, "RightClick": 1,
        "popUpWidnow": 1, "Iframe": 1, "age_of_domain": 1, "DNSRecord": 1,
        "web_traffic": 1, "Page_Rank": 1, "Google_Index": 1,
        "Links_pointing_to_page": 1, "Statistical_report": 1
    }
    
    result = client.predict(sample_features)
    print(f"预测结果: {result.prediction}")
    print(f"是否威胁: {result.is_threat}")
```

### cURL命令集合

```bash
# 1. 健康检查
curl -s http://localhost:8000/ | head -c 100

# 2. 获取特征要求
curl -s http://localhost:8000/api/features/requirements | python -m json.tool

# 3. 威胁预测
curl -X POST http://localhost:8000/predict_live \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "having_IP_Address": 1, "URL_Length": 1, "Shortining_Service": 1,
  "having_At_Symbol": 1, "double_slash_redirecting": 1, "Prefix_Suffix": -1,
  "having_Sub_Domain": 1, "SSLfinal_State": 1, "Domain_registeration_length": 1,
  "Favicon": 1, "port": 1, "HTTPS_token": 1, "Request_URL": 1,
  "URL_of_Anchor": 1, "Links_in_tags": 1, "SFH": 1, "Submitting_to_email": 1,
  "Abnormal_URL": 1, "Redirect": 0, "on_mouseover": 1, "RightClick": 1,
  "popUpWidnow": 1, "Iframe": 1, "age_of_domain": 1, "DNSRecord": 1,
  "web_traffic": 1, "Page_Rank": 1, "Google_Index": 1,
  "Links_pointing_to_page": 1, "Statistical_report": 1
}
EOF

# 4. 触发训练
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"use_deep_learning": false}'

# 5. 验证数据
curl -X POST http://localhost:8000/api/data/validate \
  -F "file=@your_data.csv"

# 6. 补全数据
curl -X POST http://localhost:8000/api/data/impute \
  -F "file=@your_data.csv" \
  -F "strategy=knn"
```

---

## 最佳实践

### 1. 性能优化

**批量预测**:
- 对于大量数据，使用 `/predict_on_test_data` 而非循环调用 `/predict_live`
- 考虑使用连接池复用HTTP连接

**训练优化**:
- 首次训练使用默认参数，验证流程正确后再调整
- 深度学习训练建议在GPU环境下进行

### 2. 错误处理

```python
import requests
from requests.exceptions import RequestException

def safe_predict(features):
    try:
        response = requests.post(
            "http://localhost:8000/predict_live",
            json=features,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return {"error": "请求超时，请稍后重试"}
    except requests.HTTPError as e:
        return {"error": f"HTTP错误: {e.response.status_code}"}
    except RequestException as e:
        return {"error": f"请求失败: {str(e)}"}
```

### 3. 数据准备建议

| 建议 | 说明 |
|------|------|
| 特征值范围 | 保持在 -1, 0, 1 范围内 |
| 数据类型 | 确保所有特征为整数类型 |
| 缺失值处理 | 使用API的补全功能或预先处理 |
| 数据量 | 训练数据建议 >1000 条 |

### 4. 安全建议

- 生产环境务必配置认证机制
- 限制CORS允许的来源
- 使用HTTPS加密传输
- 定期更新依赖包

---

## FAQ常见问题

### Q1: 为什么预测返回500错误？

**A**: 通常是因为模型文件不存在。请先执行训练：
```bash
curl -X POST http://localhost:8000/api/train
```

### Q2: 如何判断训练是否完成？

**A**: 通过WebSocket监听训练日志，当收到包含 `[FINISH]` 的消息时表示训练完成。

### Q3: 特征值必须是-1, 0, 1吗？

**A**: 这是推荐的取值范围，基于原始数据集的编码方式。其他数值也可以使用，但可能影响预测准确性。

### Q4: 如何使用自己的数据训练？

**A**: 
1. 准备CSV文件（包含30个特征）
2. 使用 `/api/data/validate` 验证数据
3. 如有缺失，使用 `/api/data/impute` 补全
4. 将补全后的数据放入 `Network_Data` 目录
5. 调用 `/api/train` 开始训练

### Q5: 深度学习和机器学习有什么区别？

**A**: 
| 对比项 | 机器学习 | 深度学习 |
|--------|----------|----------|
| 训练速度 | 快（分钟级） | 慢（小时级） |
| 数据需求 | 较少 | 较多 |
| 硬件要求 | CPU即可 | 建议GPU |
| 准确率 | 较高 | 可能更高 |

### Q6: localhost无法访问怎么办？

**A**: 尝试以下方案：
1. 使用 `127.0.0.1:8000` 替代 `localhost:8000`
2. 检查防火墙设置
3. 确认服务已启动（查看终端输出）
4. 检查端口是否被占用

### Q7: 如何查看API的交互式文档？

**A**: 访问 http://localhost:8000/docs 查看Swagger UI文档。

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v7.1.0 | 2024-12 | 添加深度学习支持、数据补全功能 |
| v7.0.0 | 2024-11 | 重构训练管道、优化WebSocket |
| v6.0.0 | 2024-10 | 添加数据验证API |

---

## 联系与支持

- **GitHub**: [Network-Security-Based-On-ML](https://github.com/zimingttkx/Network-Security-Based-On-ML)
- **Issues**: 在GitHub仓库提交Issue
- **文档**: 查看 `/wiki` 目录下的其他文档

---

> 📝 **文档维护**: 本文档随API更新同步维护，如有疑问请提交Issue。
