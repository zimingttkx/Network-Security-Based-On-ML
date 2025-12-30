# 网络安全威胁检测系统 - API接口文档

> 版本: 2.0.0 | 更新日期: 2024-12

## 目录

- [概述](#概述)
- [认证与授权](#认证与授权)
- [系统端点](#系统端点)
- [预测端点](#预测端点)
- [模型管理端点](#模型管理端点)
- [统计端点](#统计端点)
- [防火墙端点](#防火墙端点)
- [数据管理端点](#数据管理端点)
- [WebSocket端点](#websocket端点)
- [错误处理](#错误处理)
- [示例代码](#示例代码)

---

## 概述

### 基础信息

| 项目 | 值 |
|------|-----|
| 基础URL | `http://localhost:8000` |
| API版本 | v1 |
| 数据格式 | JSON |
| 字符编码 | UTF-8 |

### 请求头

```http
Content-Type: application/json
Accept: application/json
```

### 响应格式

所有API响应遵循统一格式：

```json
{
  "success": true,
  "data": { ... },
  "message": "操作成功"
}
```

错误响应：

```json
{
  "success": false,
  "error": "错误描述",
  "detail": "详细错误信息"
}
```

---

## 认证与授权

当前版本API无需认证，适用于内网部署。生产环境建议添加API Key或JWT认证。

---

## 系统端点

### 1. 健康检查

检查服务运行状态。

**请求**

```http
GET /health
```

**响应**

```json
{
  "status": "healthy",
  "timestamp": "2024-12-30T10:30:00.000000"
}
```

---

### 2. 系统统计

获取系统全局统计信息。

**请求**

```http
GET /api/system-stats
```

**响应**

```json
{
  "total_predictions": 1250,
  "correct_predictions": 1198,
  "threat_detections": 156,
  "accuracy": 95.8,
  "uptime_seconds": 86400,
  "status": "healthy",
  "version": "v2.0.0",
  "timestamp": "2024-12-30T10:30:00.000000"
}
```

**字段说明**

| 字段 | 类型 | 描述 |
|------|------|------|
| total_predictions | int | 总预测次数 |
| correct_predictions | int | 正确预测次数 |
| threat_detections | int | 威胁检测次数 |
| accuracy | float | 准确率(%) |
| uptime_seconds | int | 运行时间(秒) |

---

### 3. 记录预测

记录一次预测结果用于统计。

**请求**

```http
POST /api/record-prediction?is_correct=true&is_threat=false
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| is_correct | bool | 否 | true | 预测是否正确 |
| is_threat | bool | 否 | false | 是否为威胁 |

**响应**

```json
{
  "success": true,
  "total": 1251
}
```

---

## 预测端点

### 1. 实时威胁预测

对单个URL/请求进行威胁检测。

**请求**

```http
POST /predict_live
Content-Type: application/json

{
  "having_IP_Address": 1,
  "URL_Length": 0,
  "Shortining_Service": 1,
  "having_At_Symbol": 1,
  "double_slash_redirecting": 0,
  "Prefix_Suffix": 1,
  "having_Sub_Domain": 0,
  "SSLfinal_State": 1,
  "Domain_registeration_length": 0,
  "Favicon": 1,
  "port": 1,
  "HTTPS_token": 0,
  "Request_URL": 1,
  "URL_of_Anchor": 0,
  "Links_in_tags": 0,
  "SFH": 1,
  "Submitting_to_email": 1,
  "Abnormal_URL": 1,
  "Redirect": 0,
  "on_mouseover": 1,
  "RightClick": 1,
  "popUpWidnow": 0,
  "Iframe": 1,
  "age_of_domain": 0,
  "DNSRecord": 1,
  "web_traffic": 0,
  "Page_Rank": 0,
  "Google_Index": 1,
  "Links_pointing_to_page": 0,
  "Statistical_report": 1
}
```

**特征说明 (30个特征)**

| 特征名 | 取值 | 描述 |
|--------|------|------|
| having_IP_Address | -1/1 | URL是否包含IP地址 |
| URL_Length | -1/0/1 | URL长度 (短/中/长) |
| Shortining_Service | -1/1 | 是否使用短链接服务 |
| having_At_Symbol | -1/1 | URL是否包含@符号 |
| double_slash_redirecting | -1/1 | 是否有双斜杠重定向 |
| Prefix_Suffix | -1/1 | 域名是否有前缀/后缀 |
| having_Sub_Domain | -1/0/1 | 子域名数量 |
| SSLfinal_State | -1/0/1 | SSL证书状态 |
| Domain_registeration_length | -1/1 | 域名注册时长 |
| Favicon | -1/1 | 网站图标来源 |
| port | -1/1 | 是否使用非标准端口 |
| HTTPS_token | -1/1 | HTTPS令牌 |
| Request_URL | -1/0/1 | 请求URL比例 |
| URL_of_Anchor | -1/0/1 | 锚点URL比例 |
| Links_in_tags | -1/0/1 | 标签中链接比例 |
| SFH | -1/0/1 | 表单处理程序 |
| Submitting_to_email | -1/1 | 是否提交到邮箱 |
| Abnormal_URL | -1/1 | URL是否异常 |
| Redirect | 0/1 | 重定向次数 |
| on_mouseover | -1/1 | 鼠标悬停事件 |
| RightClick | -1/1 | 右键禁用 |
| popUpWidnow | -1/1 | 弹窗 |
| Iframe | -1/1 | iframe使用 |
| age_of_domain | -1/1 | 域名年龄 |
| DNSRecord | -1/1 | DNS记录 |
| web_traffic | -1/0/1 | 网站流量 |
| Page_Rank | -1/0/1 | 页面排名 |
| Google_Index | -1/1 | Google索引 |
| Links_pointing_to_page | -1/0/1 | 指向页面的链接数 |
| Statistical_report | -1/1 | 统计报告 |

**响应**

```json
{
  "prediction": "危险 (Malicious)",
  "raw_prediction": 1
}
```

| 字段 | 描述 |
|------|------|
| prediction | 预测结果文本 |
| raw_prediction | 原始预测值 (0=安全, 1=危险) |

---

### 2. 测试集批量预测

对测试数据集进行批量预测并生成可视化报告。

**请求**

```http
GET /predict_on_test_data
```

**响应**

```json
{
  "table_data": [
    {
      "having_IP_Address": 1,
      "URL_Length": 0,
      "prediction": "危险 (Malicious)"
    }
  ],
  "img_confusion_matrix": "data:image/png;base64,...",
  "img_pie_chart": "data:image/png;base64,..."
}
```

---

## 模型管理端点

### 1. 列出所有模型

获取系统中所有可用的检测模型。

**请求**

```http
GET /api/v1/models/list
GET /api/v1/models/list?category=ml_classifier
```

**参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| category | string | 否 | 模型类别过滤 |

**可用类别**

- `ml_classifier` - 机器学习分类器
- `dl_classifier` - 深度学习分类器
- `anomaly_detector` - 异常检测器
- `rl_agent` - 强化学习代理
- `ensemble` - 集成模型

**响应**

```json
{
  "success": true,
  "models": [
    {
      "id": "rf_nslkdd",
      "name": "Random Forest (NSL-KDD)",
      "category": "ml_classifier",
      "description": "随机森林分类器，在NSL-KDD数据集上训练",
      "attack_types": ["dos", "probe", "r2l", "u2r"],
      "accuracy": 0.956,
      "f1_score": 0.948,
      "dataset": "NSL-KDD",
      "version": "1.0",
      "is_loaded": false
    },
    {
      "id": "kitsune",
      "name": "Kitsune (KitNET)",
      "category": "anomaly_detector",
      "description": "基于AfterImage增量统计和KitNET自编码器集成的在线无监督NIDS",
      "attack_types": ["all"],
      "accuracy": 0.945,
      "f1_score": 0.932,
      "dataset": "Mirai/Custom",
      "version": "1.0",
      "is_loaded": false
    }
  ],
  "total": 15
}
```

---

### 2. 获取模型详情

**请求**

```http
GET /api/v1/models/info/{model_id}
```

**响应**

```json
{
  "success": true,
  "model": {
    "id": "kitsune",
    "name": "Kitsune (KitNET)",
    "category": "anomaly_detector",
    "description": "基于AfterImage增量统计和KitNET自编码器集成的在线无监督NIDS (NDSS'18)",
    "attack_types": ["all"],
    "accuracy": 0.945,
    "f1_score": 0.932,
    "dataset": "Mirai/Custom",
    "version": "1.0",
    "is_loaded": false
  }
}
```

---

### 3. 获取模型类别

**请求**

```http
GET /api/v1/models/categories
```

**响应**

```json
{
  "success": true,
  "categories": [
    {"id": "ml_classifier", "name": "机器学习分类器", "icon": "fa-robot"},
    {"id": "dl_classifier", "name": "深度学习分类器", "icon": "fa-brain"},
    {"id": "anomaly_detector", "name": "异常检测器", "icon": "fa-search"},
    {"id": "rl_agent", "name": "强化学习代理", "icon": "fa-gamepad"}
  ]
}
```

---

### 4. 根据攻击类型推荐模型

**请求**

```http
GET /api/v1/models/recommend?attack_type=ddos
```

**可用攻击类型**

- `dos` - DoS攻击
- `ddos` - DDoS攻击
- `probe` - 探测攻击
- `r2l` - 远程到本地攻击
- `u2r` - 用户到根攻击
- `sql_injection` - SQL注入
- `xss` - 跨站脚本
- `brute_force` - 暴力破解
- `botnet` - 僵尸网络
- `malware` - 恶意软件
- `phishing` - 钓鱼攻击
- `all` - 所有类型

**响应**

```json
{
  "success": true,
  "attack_type": "ddos",
  "recommended": [
    {
      "id": "lucid_cnn",
      "name": "LUCID CNN",
      "accuracy": 0.994
    }
  ]
}
```

---

### 5. 获取预设模型组合

**请求**

```http
GET /api/v1/models/presets
```

**响应**

```json
{
  "success": true,
  "presets": [
    {
      "id": "balanced",
      "name": "均衡防护",
      "description": "ML+DL+RL组合，平衡检测率和性能",
      "ml_models": ["rf_nslkdd", "xgb_nslkdd"],
      "dl_models": ["dnn_cicids"],
      "anomaly_models": ["isolation_forest"],
      "rl_agent": "dqn_firewall",
      "voting_strategy": "majority"
    },
    {
      "id": "high_security",
      "name": "高安全模式",
      "description": "多模型投票，最大化检测率",
      "ml_models": ["rf_nslkdd", "xgb_nslkdd"],
      "dl_models": ["dnn_cicids", "lstm_flow"],
      "anomaly_models": ["autoencoder_anomaly", "isolation_forest"],
      "rl_agent": "ppo_security",
      "voting_strategy": "any"
    }
  ]
}
```

---

### 6. 创建模型组合

**请求**

```http
POST /api/v1/models/combination
Content-Type: application/json

{
  "name": "my_custom_combo",
  "ml_models": ["rf_nslkdd", "xgb_nslkdd"],
  "dl_models": ["dnn_cicids"],
  "anomaly_models": ["kitsune"],
  "rl_agent": "dqn_security",
  "voting_strategy": "majority",
  "threshold": 0.5
}
```

**投票策略**

| 策略 | 描述 |
|------|------|
| majority | 多数投票 (过半判定威胁) |
| any | 任一检出即判定威胁 (高安全) |
| weighted | 加权投票 (按准确率) |
| all | 全部一致才判定威胁 (低误报) |

**响应**

```json
{
  "success": true,
  "message": "模型组合 'my_custom_combo' 已创建",
  "config": { ... }
}
```

---

### 7. 列出已创建的组合

**请求**

```http
GET /api/v1/models/combinations
```

---

### 8. 删除模型组合

**请求**

```http
DELETE /api/v1/models/combination/{name}
```

---

## 统计端点

### 1. 统计概览

获取流量统计概览信息。

**请求**

```http
GET /api/v1/stats/overview?hours=24
```

**参数**

| 参数 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| hours | int | 否 | 24 | 统计时间范围(1-720小时) |

**响应**

```json
{
  "total_requests": 15000,
  "blocked_requests": 1250,
  "allowed_requests": 13500,
  "challenged_requests": 250,
  "threat_counts": {
    "benign": 13500,
    "ddos": 500,
    "sql_injection": 200,
    "xss": 150
  },
  "action_counts": {
    "allow": 13500,
    "block": 1250,
    "challenge": 250
  },
  "risk_level_counts": {
    "safe": 13500,
    "low": 500,
    "medium": 600,
    "high": 300,
    "critical": 100
  },
  "top_source_ips": [
    ["192.168.1.100", 500],
    ["10.0.0.50", 300]
  ],
  "top_threat_types": [
    ["ddos", 500],
    ["sql_injection", 200]
  ],
  "avg_risk_score": 0.15,
  "avg_processing_time_ms": 12.5
}
```

---

### 2. 威胁分布

**请求**

```http
GET /api/v1/stats/threats?hours=24
```

**响应**

```json
{
  "success": true,
  "data": {
    "benign": 13500,
    "ddos": 500,
    "sql_injection": 200,
    "xss": 150,
    "brute_force": 100
  }
}
```

---

### 3. 动作分布

**请求**

```http
GET /api/v1/stats/actions?hours=24
```

---

### 4. TOP源IP

**请求**

```http
GET /api/v1/stats/sources?hours=24&limit=10&threat_only=false
```

**参数**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| hours | int | 24 | 时间范围 |
| limit | int | 10 | 返回数量(1-100) |
| threat_only | bool | false | 仅统计威胁流量 |

**响应**

```json
{
  "success": true,
  "data": [
    {"ip": "192.168.1.100", "count": 500},
    {"ip": "10.0.0.50", "count": 300}
  ]
}
```

---

### 5. 时间线统计

**请求**

```http
GET /api/v1/stats/timeline?hours=24&interval=60
```

**参数**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| hours | int | 24 | 时间范围 |
| interval | int | 60 | 时间间隔(分钟, 5-1440) |

---

### 6. 地理位置分布

**请求**

```http
GET /api/v1/stats/geo?hours=24
```

---

### 7. 最近威胁

**请求**

```http
GET /api/v1/stats/recent-threats?limit=20
```

---

### 8. 风险分数分布

**请求**

```http
GET /api/v1/stats/risk-distribution?hours=24&bins=10
```

---

### 9. 查询流量日志

**请求**

```http
GET /api/v1/stats/logs?hours=24&source_ip=192.168.1.1&threat_type=ddos&action=block&min_risk_score=0.5&limit=100&offset=0
```

**参数**

| 参数 | 类型 | 描述 |
|------|------|------|
| hours | int | 时间范围 |
| source_ip | string | 源IP过滤 |
| threat_type | string | 威胁类型过滤 |
| action | string | 动作过滤 |
| min_risk_score | float | 最小风险分数 |
| limit | int | 返回数量 |
| offset | int | 偏移量 |

---

### 10. 创建流量日志

**请求**

```http
POST /api/v1/stats/logs
Content-Type: application/json

{
  "source_ip": "192.168.1.100",
  "source_port": 54321,
  "dest_ip": "10.0.0.1",
  "dest_port": 80,
  "protocol": "HTTP",
  "method": "GET",
  "url": "/api/login",
  "user_agent": "Mozilla/5.0",
  "threat_type": "brute_force",
  "risk_level": "high",
  "risk_score": 0.85,
  "action": "block",
  "geo_country": "CN",
  "geo_city": "Beijing",
  "processing_time_ms": 15.5
}
```

**威胁类型 (threat_type)**

- `benign` - 正常流量
- `ddos` - DDoS攻击
- `dos` - DoS攻击
- `sql_injection` - SQL注入
- `xss` - 跨站脚本
- `brute_force` - 暴力破解
- `port_scan` - 端口扫描
- `malware` - 恶意软件
- `botnet` - 僵尸网络
- `phishing` - 钓鱼攻击

**风险等级 (risk_level)**

- `safe` - 安全
- `low` - 低风险
- `medium` - 中风险
- `high` - 高风险
- `critical` - 严重

**动作 (action)**

- `allow` - 允许
- `block` - 阻止
- `challenge` - 验证
- `rate_limit` - 限流
- `log` - 仅记录

---

### 11. 清理旧日志

**请求**

```http
DELETE /api/v1/stats/logs/cleanup?days=30
```

---

### 12. 生成演示数据

**请求**

```http
POST /api/v1/stats/demo-data?count=500&hours_back=24
```

---

## 防火墙端点

### 1. 威胁检测

对单个请求进行威胁检测。

**请求**

```http
POST /api/v1/firewall/detect
Content-Type: application/json

{
  "features": {
    "packet_rate": 1000.5,
    "byte_rate": 50000.0,
    "connection_count": 150,
    "src_port_entropy": 0.8,
    "dst_port_entropy": 0.2,
    "protocol_type": 6,
    "flag_count": 3
  },
  "model_name": "kitsune"
}
```

**响应**

```json
{
  "is_threat": true,
  "threat_level": "high",
  "confidence": 0.92,
  "action": "block",
  "model_name": "kitsune",
  "threat_type": "ddos",
  "detection_time": 0.015
}
```

**威胁等级 (threat_level)**

- `safe` - 安全
- `low` - 低威胁
- `medium` - 中威胁
- `high` - 高威胁
- `critical` - 严重威胁

**动作 (action)**

- `allow` - 允许通过
- `block` - 阻止
- `challenge` - 需要验证
- `rate_limit` - 限制速率
- `quarantine` - 隔离
- `alert` - 告警
- `log` - 仅记录

---

### 2. 批量威胁检测

**请求**

```http
POST /api/v1/firewall/detect/batch
Content-Type: application/json

{
  "features_list": [
    {"packet_rate": 1000.5, "byte_rate": 50000.0},
    {"packet_rate": 100.0, "byte_rate": 5000.0}
  ],
  "model_name": null
}
```

---

### 3. 列出检测模型

**请求**

```http
GET /api/v1/firewall/models
```

---

### 4. 生成验证码

**请求**

```http
POST /api/v1/firewall/captcha/generate
Content-Type: application/json

{
  "challenge_type": "math"
}
```

**验证码类型**

- `math` - 数学计算题
- `text` - 文本验证

**响应**

```json
{
  "challenge_id": "abc123",
  "challenge_type": "math",
  "question": "15 + 27 = ?",
  "expires_at": 1703923200.0
}
```

---

### 5. 验证答案

**请求**

```http
POST /api/v1/firewall/captcha/verify
Content-Type: application/json

{
  "challenge_id": "abc123",
  "answer": "42"
}
```

**响应**

```json
{
  "success": true,
  "message": "验证成功"
}
```

---

### 6. 验证码统计

**请求**

```http
GET /api/v1/firewall/captcha/stats
```

---

### 7. 防火墙健康检查

**请求**

```http
GET /api/v1/firewall/health
```

---

## 数据管理端点

### 1. 获取特征要求

**请求**

```http
GET /api/features/requirements
```

---

### 2. 验证数据文件

**请求**

```http
POST /api/data/validate
Content-Type: multipart/form-data

file: <CSV文件>
```

**响应**

```json
{
  "status": "success",
  "filename": "data.csv",
  "rows": 1000,
  "columns": 31,
  "is_valid": true,
  "validation_report": {
    "missing_features": [],
    "extra_features": [],
    "invalid_values": []
  },
  "imputation_suggestions": {}
}
```

---

### 3. 补全数据特征

**请求**

```http
POST /api/data/impute
Content-Type: multipart/form-data

file: <CSV文件>
strategy: constant
fill_value: 0
```

**补全策略**

- `constant` - 常量填充
- `mean` - 均值填充
- `median` - 中位数填充
- `most_frequent` - 众数填充

---

### 4. 下载补全后的数据

**请求**

```http
GET /api/data/download/{filename}
```

---

## 训练端点

### 1. 触发模型训练

**请求**

```http
POST /api/train
Content-Type: application/json

{
  "use_deep_learning": false,
  "dl_model_type": "dnn",
  "dl_config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**深度学习模型类型**

- `dnn` - 深度神经网络
- `cnn` - 卷积神经网络
- `lstm` - 长短期记忆网络

**响应**

```json
{
  "status": "success",
  "message": "机器学习模型训练任务已在后台启动"
}
```

---

## WebSocket端点

### 训练日志实时推送

**连接**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/train');

ws.onmessage = function(event) {
  console.log('训练日志:', event.data);
};

ws.onopen = function() {
  console.log('WebSocket连接已建立');
};
```

---

## 错误处理

### HTTP状态码

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

---

## 示例代码

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. 实时预测
features = {
    "having_IP_Address": 1,
    "URL_Length": 0,
    # ... 其他30个特征
}
response = requests.post(f"{BASE_URL}/predict_live", json=features)
print(response.json())

# 3. 威胁检测
detect_data = {
    "features": {"packet_rate": 1000.5, "byte_rate": 50000.0},
    "model_name": "kitsune"
}
response = requests.post(f"{BASE_URL}/api/v1/firewall/detect", json=detect_data)
print(response.json())

# 4. 获取统计概览
response = requests.get(f"{BASE_URL}/api/v1/stats/overview?hours=24")
print(response.json())
```

### cURL

```bash
# 健康检查
curl http://localhost:8000/health

# 获取模型列表
curl http://localhost:8000/api/v1/models/list

# 威胁检测
curl -X POST http://localhost:8000/api/v1/firewall/detect \
  -H "Content-Type: application/json" \
  -d '{"features": {"packet_rate": 1000.5}, "model_name": "kitsune"}'

# 获取统计
curl "http://localhost:8000/api/v1/stats/overview?hours=24"
```

### JavaScript

```javascript
// 使用Fetch API
async function detectThreat(features) {
  const response = await fetch('http://localhost:8000/api/v1/firewall/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features, model_name: 'kitsune' })
  });
  return response.json();
}

// 使用示例
detectThreat({ packet_rate: 1000.5, byte_rate: 50000.0 })
  .then(result => console.log(result));
```

---

## 附录

### A. 完整特征列表

| 序号 | 特征名 | 类型 | 取值范围 |
|------|--------|------|----------|
| 1 | having_IP_Address | int | -1, 1 |
| 2 | URL_Length | int | -1, 0, 1 |
| 3 | Shortining_Service | int | -1, 1 |
| 4 | having_At_Symbol | int | -1, 1 |
| 5 | double_slash_redirecting | int | -1, 1 |
| 6 | Prefix_Suffix | int | -1, 1 |
| 7 | having_Sub_Domain | int | -1, 0, 1 |
| 8 | SSLfinal_State | int | -1, 0, 1 |
| 9 | Domain_registeration_length | int | -1, 1 |
| 10 | Favicon | int | -1, 1 |
| 11 | port | int | -1, 1 |
| 12 | HTTPS_token | int | -1, 1 |
| 13 | Request_URL | int | -1, 0, 1 |
| 14 | URL_of_Anchor | int | -1, 0, 1 |
| 15 | Links_in_tags | int | -1, 0, 1 |
| 16 | SFH | int | -1, 0, 1 |
| 17 | Submitting_to_email | int | -1, 1 |
| 18 | Abnormal_URL | int | -1, 1 |
| 19 | Redirect | int | 0, 1 |
| 20 | on_mouseover | int | -1, 1 |
| 21 | RightClick | int | -1, 1 |
| 22 | popUpWidnow | int | -1, 1 |
| 23 | Iframe | int | -1, 1 |
| 24 | age_of_domain | int | -1, 1 |
| 25 | DNSRecord | int | -1, 1 |
| 26 | web_traffic | int | -1, 0, 1 |
| 27 | Page_Rank | int | -1, 0, 1 |
| 28 | Google_Index | int | -1, 1 |
| 29 | Links_pointing_to_page | int | -1, 0, 1 |
| 30 | Statistical_report | int | -1, 1 |

### B. 模型性能对比

| 模型 | 准确率 | F1分数 | 适用场景 |
|------|--------|--------|----------|
| Random Forest | 95.6% | 94.8% | 通用检测 |
| XGBoost | 96.2% | 95.5% | 高精度检测 |
| Kitsune | 94.5% | 93.2% | 在线异常检测 |
| LUCID CNN | 99.4% | 99.3% | DDoS专项检测 |
| Slips | 91.2% | 89.5% | 行为分析 |
| DQN Agent | 93.5% | 92.8% | 智能响应 |
| PPO Agent | 94.8% | 94.1% | 平衡决策 |

---

*文档版本: 2.0.0 | 最后更新: 2024-12*
