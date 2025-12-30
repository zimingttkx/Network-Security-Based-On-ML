# 网络安全威胁检测系统 - 服务器部署与网站防护指南

> 版本: 2.0.0 | 更新日期: 2024-12

本指南详细介绍如何在生产服务器上部署网络安全威胁检测系统，并将其集成到您的网站防护体系中。

---

## 目录

- [系统要求](#系统要求)
- [快速部署](#快速部署)
- [Docker部署](#docker部署)
- [Kubernetes部署](#kubernetes部署)
- [Nginx反向代理配置](#nginx反向代理配置)
- [网站防护集成](#网站防护集成)
- [性能优化](#性能优化)
- [监控与告警](#监控与告警)
- [安全加固](#安全加固)
- [故障排除](#故障排除)

---

## 系统要求

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 2核 | 4核+ |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 20GB SSD | 50GB+ SSD |
| 网络 | 100Mbps | 1Gbps |

### 软件要求

- **操作系统**: Ubuntu 20.04/22.04 LTS, CentOS 7/8, Debian 10/11
- **Python**: 3.10+
- **Docker**: 20.10+ (可选)
- **Kubernetes**: 1.24+ (可选)

---

## 快速部署

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 启动服务

```bash
# 开发模式
python app.py

# 生产模式 (使用Gunicorn)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

### 5. 验证部署

```bash
curl http://localhost:8000/health
# 预期输出: {"status":"healthy","timestamp":"..."}
```

---

## Docker部署

### 1. 构建镜像

```bash
docker build -t network-security:latest .
```

### 2. 运行容器

```bash
# 基础运行
docker run -d \
  --name network-security \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  network-security:latest

# 带资源限制
docker run -d \
  --name network-security \
  -p 8000:8000 \
  --memory="4g" \
  --cpus="2" \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --restart=unless-stopped \
  network-security:latest
```

### 3. Docker Compose部署

```bash
docker-compose up -d
```

**docker-compose.yml 配置说明:**

```yaml
version: '3.8'
services:
  network-security:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - WORKERS=4
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 4. 查看日志

```bash
docker logs -f network-security
```

---

## Kubernetes部署

### 1. 创建命名空间

```bash
kubectl create namespace network-security
```

### 2. 部署应用

```bash
kubectl apply -f deploy/kubernetes/ -n network-security
```

### 3. 检查部署状态

```bash
kubectl get pods -n network-security
kubectl get svc -n network-security
```

### 4. 扩展副本

```bash
kubectl scale deployment network-security --replicas=3 -n network-security
```

### 5. 配置HPA自动扩缩

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: network-security-hpa
  namespace: network-security
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: network-security
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Nginx反向代理配置

### 1. 安装Nginx

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install nginx -y

# CentOS/RHEL
sudo yum install nginx -y
```

### 2. 配置反向代理

创建 `/etc/nginx/sites-available/network-security`:

```nginx
upstream network_security {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name security.yourdomain.com;
    
    # 重定向到HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name security.yourdomain.com;
    
    # SSL证书
    ssl_certificate /etc/letsencrypt/live/security.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/security.yourdomain.com/privkey.pem;
    
    # SSL配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # 日志
    access_log /var/log/nginx/network-security-access.log;
    error_log /var/log/nginx/network-security-error.log;
    
    # API代理
    location /api/ {
        proxy_pass http://network_security;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # WebSocket代理
    location /ws/ {
        proxy_pass http://network_security;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
    
    # 静态文件
    location /static/ {
        alias /path/to/Network-Security-Based-On-ML/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
    
    # 前端页面
    location / {
        proxy_pass http://network_security;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. 启用配置

```bash
sudo ln -s /etc/nginx/sites-available/network-security /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 网站防护集成

### 方案一: Nginx Lua模块集成

在Nginx中使用Lua脚本调用检测API：

```nginx
# 安装OpenResty或nginx-lua-module

location / {
    access_by_lua_block {
        local http = require "resty.http"
        local httpc = http.new()
        
        -- 提取请求特征
        local features = {
            source_ip = ngx.var.remote_addr,
            url = ngx.var.request_uri,
            method = ngx.var.request_method,
            user_agent = ngx.var.http_user_agent or ""
        }
        
        -- 调用检测API
        local res, err = httpc:request_uri("http://127.0.0.1:8000/api/v1/firewall/detect", {
            method = "POST",
            body = require("cjson").encode({features = features}),
            headers = {["Content-Type"] = "application/json"}
        })
        
        if res and res.status == 200 then
            local result = require("cjson").decode(res.body)
            if result.is_threat and result.action == "block" then
                ngx.exit(ngx.HTTP_FORBIDDEN)
            end
        end
    }
    
    proxy_pass http://your_backend;
}
```

### 方案二: 应用层中间件集成

#### Python Flask/FastAPI

```python
import requests
from functools import wraps
from flask import request, abort

SECURITY_API = "http://localhost:8000"

def security_check(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 提取请求特征
        features = {
            "source_ip": request.remote_addr,
            "url": request.url,
            "method": request.method,
            "user_agent": request.headers.get("User-Agent", "")
        }
        
        # 调用检测API
        try:
            response = requests.post(
                f"{SECURITY_API}/api/v1/firewall/detect",
                json={"features": features},
                timeout=1
            )
            result = response.json()
            
            if result.get("is_threat") and result.get("action") == "block":
                abort(403, "Access Denied - Threat Detected")
        except Exception as e:
            # 检测服务不可用时放行（fail-open）
            pass
        
        return f(*args, **kwargs)
    return decorated_function

# 使用示例
@app.route("/api/sensitive")
@security_check
def sensitive_endpoint():
    return {"data": "sensitive"}
```

#### Node.js Express

```javascript
const axios = require('axios');

const SECURITY_API = 'http://localhost:8000';

const securityMiddleware = async (req, res, next) => {
  const features = {
    source_ip: req.ip,
    url: req.originalUrl,
    method: req.method,
    user_agent: req.get('User-Agent') || ''
  };

  try {
    const response = await axios.post(
      `${SECURITY_API}/api/v1/firewall/detect`,
      { features },
      { timeout: 1000 }
    );

    if (response.data.is_threat && response.data.action === 'block') {
      return res.status(403).json({ error: 'Access Denied' });
    }
  } catch (error) {
    // Fail-open: 检测服务不可用时放行
    console.error('Security check failed:', error.message);
  }

  next();
};

// 使用示例
app.use('/api', securityMiddleware);
```

### 方案三: 独立WAF模式

将系统部署为独立WAF，所有流量先经过检测：

```
用户请求 → Nginx → 安全检测API → 后端服务
                      ↓
                  阻止/放行
```

**Nginx配置:**

```nginx
location / {
    # 先发送到检测服务
    auth_request /security-check;
    auth_request_set $security_action $upstream_http_x_security_action;
    
    # 根据检测结果处理
    if ($security_action = "block") {
        return 403;
    }
    
    proxy_pass http://backend;
}

location = /security-check {
    internal;
    proxy_pass http://127.0.0.1:8000/api/v1/firewall/detect;
    proxy_pass_request_body on;
    proxy_set_header Content-Type "application/json";
    proxy_set_header X-Original-URI $request_uri;
    proxy_set_header X-Real-IP $remote_addr;
}
```

---

## 实时流量监控与防护

### 1. 配置流量日志记录

在您的应用中添加流量日志记录：

```python
import requests
from datetime import datetime

def log_traffic(request, response, is_threat=False, threat_type="benign"):
    """记录流量到安全系统"""
    log_data = {
        "source_ip": request.remote_addr,
        "source_port": request.environ.get('REMOTE_PORT', 0),
        "dest_ip": request.host.split(':')[0],
        "dest_port": int(request.host.split(':')[1]) if ':' in request.host else 80,
        "protocol": "HTTPS" if request.is_secure else "HTTP",
        "method": request.method,
        "url": request.url,
        "user_agent": request.headers.get("User-Agent", ""),
        "threat_type": threat_type,
        "risk_level": "high" if is_threat else "safe",
        "risk_score": 0.9 if is_threat else 0.1,
        "action": "block" if is_threat else "allow",
        "processing_time_ms": response.elapsed.total_seconds() * 1000
    }
    
    try:
        requests.post(
            "http://localhost:8000/api/v1/stats/logs",
            json=log_data,
            timeout=1
        )
    except:
        pass  # 日志记录失败不影响主业务
```

### 2. 实时仪表盘监控

访问 `http://your-server:8000/dashboard` 查看实时监控仪表盘，包括：

- 总请求数和威胁检测数
- 威胁类型分布饼图
- 时间线趋势图
- TOP攻击源IP
- 最近威胁列表

### 3. 设置告警规则

```python
import requests
import smtplib
from email.mime.text import MIMEText

def check_and_alert():
    """检查威胁并发送告警"""
    response = requests.get(
        "http://localhost:8000/api/v1/stats/overview?hours=1"
    )
    stats = response.json()
    
    # 告警阈值
    THREAT_THRESHOLD = 100  # 1小时内威胁数超过100
    BLOCK_RATE_THRESHOLD = 0.1  # 阻止率超过10%
    
    total = stats["total_requests"]
    blocked = stats["blocked_requests"]
    
    if blocked > THREAT_THRESHOLD or (total > 0 and blocked/total > BLOCK_RATE_THRESHOLD):
        send_alert(f"安全告警: 检测到 {blocked} 个威胁，阻止率 {blocked/total*100:.1f}%")

def send_alert(message):
    """发送告警邮件"""
    msg = MIMEText(message)
    msg['Subject'] = '网络安全告警'
    msg['From'] = 'security@yourdomain.com'
    msg['To'] = 'admin@yourdomain.com'
    
    with smtplib.SMTP('smtp.yourdomain.com') as server:
        server.send_message(msg)
```

---

## 性能优化

### 1. 启用多进程

```bash
# Gunicorn多进程
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  app:app
```

### 2. 配置Redis缓存

```python
# 安装redis
# pip install redis

import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_detect(features, ttl=60):
    """带缓存的威胁检测"""
    # 生成缓存键
    cache_key = f"detect:{hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()}"
    
    # 尝试从缓存获取
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 调用检测API
    response = requests.post(
        "http://localhost:8000/api/v1/firewall/detect",
        json={"features": features}
    )
    result = response.json()
    
    # 缓存结果
    redis_client.setex(cache_key, ttl, json.dumps(result))
    
    return result
```

### 3. 异步检测

```python
import asyncio
import aiohttp

async def async_detect(features):
    """异步威胁检测"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/firewall/detect",
            json={"features": features}
        ) as response:
            return await response.json()

# 批量异步检测
async def batch_async_detect(features_list):
    tasks = [async_detect(f) for f in features_list]
    return await asyncio.gather(*tasks)
```

### 4. 连接池配置

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    """创建带连接池的会话"""
    session = requests.Session()
    
    retry = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    )
    
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=100,
        max_retries=retry
    )
    
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

# 全局会话
security_session = create_session()
```

---

## 监控与告警

### 1. Prometheus指标

添加Prometheus指标导出：

```python
# pip install prometheus-client

from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# 定义指标
REQUEST_COUNT = Counter('security_requests_total', 'Total requests', ['status'])
THREAT_COUNT = Counter('security_threats_total', 'Total threats detected', ['type'])
DETECTION_TIME = Histogram('security_detection_seconds', 'Detection time')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. Grafana仪表盘

导入预配置的Grafana仪表盘：

```json
{
  "dashboard": {
    "title": "Network Security Dashboard",
    "panels": [
      {
        "title": "Requests per Second",
        "type": "graph",
        "targets": [
          {"expr": "rate(security_requests_total[5m])"}
        ]
      },
      {
        "title": "Threats Detected",
        "type": "stat",
        "targets": [
          {"expr": "sum(security_threats_total)"}
        ]
      }
    ]
  }
}
```

### 3. 日志聚合 (ELK Stack)

配置Filebeat收集日志：

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    paths:
      - /var/log/network-security/*.log
    json.keys_under_root: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "network-security-%{+yyyy.MM.dd}"
```

---

## 安全加固

### 1. API认证

添加API Key认证：

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# 使用
@app.post("/api/v1/firewall/detect")
async def detect(request: DetectRequest, api_key: str = Security(verify_api_key)):
    ...
```

### 2. 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/firewall/detect")
@limiter.limit("100/minute")
async def detect(request: Request):
    ...
```

### 3. IP白名单

```python
ALLOWED_IPS = ["10.0.0.0/8", "192.168.0.0/16", "127.0.0.1"]

from ipaddress import ip_address, ip_network

def check_ip_allowed(client_ip: str) -> bool:
    ip = ip_address(client_ip)
    return any(ip in ip_network(net) for net in ALLOWED_IPS)

@app.middleware("http")
async def ip_whitelist_middleware(request: Request, call_next):
    if not check_ip_allowed(request.client.host):
        return JSONResponse(status_code=403, content={"error": "IP not allowed"})
    return await call_next(request)
```

### 4. HTTPS强制

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# 生产环境启用
if os.getenv("ENV") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
```

---

## 故障排除

### 常见问题

#### 1. 服务无法启动

```bash
# 检查端口占用
lsof -i :8000

# 检查日志
tail -f logs/app.log

# 检查依赖
pip check
```

#### 2. 检测延迟高

```bash
# 检查系统资源
top -p $(pgrep -f "python app.py")

# 检查模型加载
curl http://localhost:8000/api/v1/models/list

# 增加worker数量
gunicorn -w 8 ...
```

#### 3. 内存占用过高

```bash
# 限制内存
docker run --memory="4g" ...

# 或在systemd中限制
# MemoryLimit=4G
```

#### 4. 模型预测错误

```bash
# 检查模型文件
ls -la models/

# 重新训练模型
curl -X POST http://localhost:8000/api/train
```

### 日志级别调整

```python
# 设置环境变量
export LOG_LEVEL=DEBUG

# 或在代码中
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

ENDPOINT="http://localhost:8000/health"
TIMEOUT=5

response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT $ENDPOINT)

if [ "$response" = "200" ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP $response)"
    exit 1
fi
```

---

## 附录

### A. Systemd服务配置

创建 `/etc/systemd/system/network-security.service`:

```ini
[Unit]
Description=Network Security Detection Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/network-security
Environment=PATH=/opt/network-security/venv/bin
ExecStart=/opt/network-security/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable network-security
sudo systemctl start network-security
```

### B. 定时任务配置

```bash
# crontab -e

# 每小时清理旧日志
0 * * * * curl -X DELETE "http://localhost:8000/api/v1/stats/logs/cleanup?days=7"

# 每天凌晨备份模型
0 2 * * * tar -czf /backup/models-$(date +\%Y\%m\%d).tar.gz /opt/network-security/models/

# 每5分钟健康检查
*/5 * * * * /opt/network-security/health_check.sh || systemctl restart network-security
```

### C. 环境变量配置

```bash
# .env 文件
WORKERS=4
LOG_LEVEL=info
MODEL_PATH=/opt/network-security/models
REDIS_URL=redis://localhost:6379/0
API_KEY=your-secret-key
ALLOWED_ORIGINS=https://yourdomain.com
```

---

*文档版本: 2.0.0 | 最后更新: 2024-12*
