# API文档

## 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

## 完整API文档

访问 http://localhost:8000/docs 查看交互式API文档（Swagger UI）。

## 核心端点

### 1. 威胁预测

#### POST /predict_live

预测网络流量是否存在威胁。

**请求体**:
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

**响应**:
```json
{
  "prediction": "安全",
  "raw_prediction": 0
}
```

**状态码**:
- `200`: 成功
- `422`: 参数错误
- `500`: 服务器错误

### 2. 模型训练

#### POST /api/train

触发模型训练任务。

**响应**:
```json
{
  "status": "success",
  "message": "模型训练任务已在后台启动"
}
```

### 3. 特征要求

#### GET /api/features/requirements

获取模型所需的30个特征信息。

**响应**:
```json
{
  "total_features": 30,
  "features": [
    {
      "name": "having_IP_Address",
      "description": "URL中是否包含IP地址 (-1: 是, 1: 否)",
      "type": "integer",
      "typical_values": "-1, 0, 1"
    },
    ...
  ]
}
```

### 4. 数据验证

#### POST /api/data/validate

验证上传的CSV数据文件。

**请求**:
- Content-Type: `multipart/form-data`
- 参数: `file` (CSV文件)

**响应**:
```json
{
  "status": "success",
  "filename": "data.csv",
  "rows": 1000,
  "columns": 25,
  "is_valid": false,
  "validation_report": {
    "missing_features": ["age_of_domain", "DNSRecord"],
    "missing_values": {},
    "recommendations": []
  }
}
```

### 5. 数据补全

#### POST /api/data/impute

补全缺失的特征。

**请求**:
- Content-Type: `multipart/form-data`
- 参数:
  - `file`: CSV文件
  - `strategy`: 补全策略 (mean/median/most_frequent/constant/knn)
  - `fill_value`: 填充值（可选）

**响应**:
```json
{
  "status": "success",
  "message": "数据补全成功",
  "output_file": "uploads/imputed_data.csv",
  "impute_report": {
    "added_features": ["age_of_domain", "DNSRecord"],
    "imputed_values": {}
  }
}
```

### 6. 下载补全数据

#### GET /api/data/download/{filename}

下载补全后的数据文件。

**参数**:
- `filename`: 文件名

**响应**: CSV文件下载

## Python客户端示例

### 预测示例

```python
import requests

url = "http://localhost:8000/predict_live"
data = {
    "having_IP_Address": 1,
    "URL_Length": 1,
    # ... 其他特征
}

response = requests.post(url, json=data)
result = response.json()

print(f"预测结果: {result['prediction']}")
```

### 训练示例

```python
import requests

url = "http://localhost:8000/api/train"
response = requests.post(url)
result = response.json()

print(result['message'])
```

### 数据验证示例

```python
import requests

url = "http://localhost:8000/api/data/validate"
files = {'file': open('data.csv', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"验证状态: {result['is_valid']}")
```

## 错误处理

所有API在出错时返回以下格式：

```json
{
  "detail": "错误描述信息"
}
```

常见错误码：
- `400`: 请求参数错误
- `404`: 资源不存在
- `422`: 数据验证失败
- `500`: 服务器内部错误

## 速率限制

目前没有速率限制，但建议合理使用API。

## WebSocket

### 训练日志推送

连接到 `ws://localhost:8000/ws/train` 可以实时接收训练日志。

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/train');

ws.onmessage = function(event) {
    console.log('训练日志:', event.data);
};
```

## 注意事项

1. 所有特征值必须为数值类型
2. 特征值通常为 -1, 0, 1
3. 必须提供全部30个特征
4. 训练任务在后台异步执行
5. 大文件上传可能需要较长时间
