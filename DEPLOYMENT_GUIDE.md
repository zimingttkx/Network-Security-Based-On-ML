# 🚀 部署配置指南

## 📋 目录

- [本地开发环境](#本地开发环境)
- [Docker部署](#docker部署)
- [云服务器部署](#云服务器部署)
- [Kubernetes部署](#kubernetes部署)
- [域名和SSL配置](#域名和ssl配置)
- [常见问题](#常见问题)

---

## 🏠 本地开发环境

### 配置步骤

#### 1. 复制环境变量文件
```bash
cp .env.example .env
```

#### 2. 编辑 .env 文件
```bash
# 本地MongoDB
MONGO_DB_URL=mongodb://localhost:27017/networksecurity

# 应用配置
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development

# MLflow
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

#### 3. 启动应用
```bash
# 启动MongoDB（如果使用本地MongoDB）
mongod

# 启动MLflow（可选）
mlflow ui --port 5000

# 启动应用
python -m networksecurity.api.app
# 或
uvicorn networksecurity.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. 访问地址

| 服务 | 本地访问地址 | 说明 |
|-----|-------------|------|
| API文档 | http://127.0.0.1:8000/api/docs | Swagger UI |
| 健康检查 | http://127.0.0.1:8000/health | 健康状态 |
| Metrics | http://127.0.0.1:8000/metrics | Prometheus指标 |
| MLflow | http://127.0.0.1:5000 | 实验追踪 |

**局域网访问：**
如果需要从局域网其他设备访问：
1. 查看本机IP: `ipconfig getifaddr en0` (Mac) 或 `ipconfig` (Windows)
2. 使用 `http://your-local-ip:8000` 访问

---

## 🐳 Docker部署

### 单容器部署

#### 1. 配置环境变量
```bash
cp .env.example .env
vim .env
```

修改为：
```bash
MONGO_DB_URL=mongodb://your-mongodb-server:27017/networksecurity
APP_HOST=0.0.0.0
APP_PORT=8000
```

#### 2. 构建镜像
```bash
docker build -t network-security-api:latest .
```

#### 3. 运行容器
```bash
docker run -d \
  --name network-security-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/final_models:/app/final_models \
  network-security-api:latest
```

#### 4. 访问地址

| 服务 | 访问地址 | 说明 |
|-----|---------|------|
| API | http://服务器IP:8000 | 主应用 |
| API文档 | http://服务器IP:8000/api/docs | Swagger |
| 健康检查 | http://服务器IP:8000/health | 健康状态 |

### Docker Compose部署（推荐）

#### 1. 配置环境变量
```bash
cp .env.example .env
vim .env
```

#### 2. 启动所有服务
```bash
docker-compose up -d
```

#### 3. 访问地址

| 服务 | 访问地址 | 默认端口 | 说明 |
|-----|---------|---------|------|
| API | http://服务器IP:8000 | 8000 | 主应用 |
| MongoDB | mongodb://服务器IP:27017 | 27017 | 数据库 |
| Prometheus | http://服务器IP:9090 | 9090 | 监控 |
| Grafana | http://服务器IP:3000 | 3000 | 可视化 |
| Redis | redis://服务器IP:6379 | 6379 | 缓存 |
| Nginx | http://服务器IP:80 | 80 | 反向代理 |

#### 4. 配置防火墙
```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp
sudo ufw allow 9090/tcp
sudo ufw allow 3000/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --add-port=9090/tcp --permanent
sudo firewall-cmd --add-port=3000/tcp --permanent
sudo firewall-cmd --reload
```

---

## ☁️ 云服务器部署

### AWS EC2

#### 1. 获取服务器公网IP
```bash
curl http://169.254.169.254/latest/meta-data/public-ipv4
```

#### 2. 配置安全组
在AWS控制台添加入站规则：
- 类型: 自定义TCP
- 端口: 8000, 9090, 3000
- 源: 0.0.0.0/0 (或特定IP)

#### 3. 部署应用
```bash
# 克隆项目
git clone https://github.com/your-username/network-security.git
cd network-security

# 配置环境变量
cp .env.example .env
vim .env

# 使用Docker Compose部署
docker-compose up -d
```

#### 4. 访问地址
```
http://公网IP:8000/api/docs
```

### 阿里云ECS

#### 1. 获取公网IP
```bash
curl https://myip.ipip.net
```

#### 2. 配置安全组
在阿里云控制台添加安全组规则：
- 端口范围: 8000/8000, 9090/9090, 3000/3000
- 授权对象: 0.0.0.0/0

#### 3. 部署流程
同AWS EC2

### 腾讯云CVM

配置步骤类似，在安全组中开放相应端口。

---

## ☸️ Kubernetes部署

### 1. 创建命名空间
```bash
kubectl create namespace production
```

### 2. 配置Secrets
```bash
# 编辑secrets配置
cp deployment/kubernetes/secrets.yaml.example deployment/kubernetes/secrets.yaml

# 修改MongoDB URL等敏感信息
vim deployment/kubernetes/secrets.yaml

# 应用配置
kubectl apply -f deployment/kubernetes/secrets.yaml
```

### 3. 修改配置中的镜像地址
```bash
vim deployment/kubernetes/deployment.yaml
```

修改镜像地址为你的Docker仓库：
```yaml
image: your-registry.com/network-security-api:latest
```

### 4. 部署应用
```bash
kubectl apply -f deployment/kubernetes/
```

### 5. 获取访问地址

#### LoadBalancer类型
```bash
kubectl get svc network-security-api-service -n production

# 输出示例：
# NAME                              TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)        AGE
# network-security-api-service      LoadBalancer   10.0.0.1        52.12.34.56     80:30000/TCP   5m
```

访问地址: `http://EXTERNAL-IP/api/docs`

#### NodePort类型
如果使用NodePort：
```bash
kubectl get svc network-security-api-service -n production

# 获取NodePort端口（例如：30123）
# 访问地址: http://任意节点IP:30123/api/docs
```

### 6. 配置Ingress（推荐）

创建 `ingress.yaml`：
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: network-security-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: network-security-api-service
            port:
              number: 80
```

应用：
```bash
kubectl apply -f ingress.yaml
```

访问地址: `http://api.yourcompany.com/api/docs`

---

## 🌐 域名和SSL配置

### 使用Nginx反向代理

#### 1. 安装Nginx
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

#### 2. 配置Nginx
```bash
sudo vim /etc/nginx/sites-available/network-security
```

基本配置：
```nginx
server {
    listen 80;
    server_name api.yourcompany.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/network-security /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 3. 配置SSL证书（Let's Encrypt）

```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d api.yourcompany.com

# 自动续期
sudo certbot renew --dry-run
```

访问地址: `https://api.yourcompany.com/api/docs`

### 使用Cloudflare

1. 添加域名到Cloudflare
2. 配置DNS记录：
   - 类型: A
   - 名称: api
   - 内容: 服务器IP
   - 代理状态: 已代理（橙色云朵）
3. SSL/TLS设置: 完全(严格)

访问地址: `https://api.yourcompany.com/api/docs`

---

## 🔧 环境配置示例

### 开发环境
```bash
# .env
MONGO_DB_URL=mongodb://localhost:27017/networksecurity
APP_ENV=development
APP_DEBUG=true
APP_HOST=0.0.0.0
APP_PORT=8000
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

访问: `http://127.0.0.1:8000`

### 测试环境
```bash
# .env
MONGO_DB_URL=mongodb://test-db-server:27017/networksecurity
APP_ENV=staging
APP_DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000
MLFLOW_TRACKING_URI=http://mlflow-test:5000
```

访问: `http://test-server-ip:8000`

### 生产环境
```bash
# .env
MONGO_DB_URL=mongodb+srv://user:pass@prod-cluster.mongodb.net/networksecurity
APP_ENV=production
APP_DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000
MLFLOW_TRACKING_URI=http://mlflow-prod:5000
API_KEY=your_secure_api_key
```

访问: `https://api.yourcompany.com`

---

## ❓ 常见问题

### Q1: localhost无法访问？

**A:** 使用 `0.0.0.0` 代替 `localhost`，或使用服务器的实际IP地址。

```bash
# 错误
uvicorn app:app --host localhost

# 正确
uvicorn app:app --host 0.0.0.0
```

### Q2: 外网无法访问？

**A:** 检查以下几点：
1. 应用是否绑定到 `0.0.0.0`
2. 防火墙是否开放端口
3. 云服务器安全组是否配置
4. 容器端口是否正确映射

### Q3: Docker容器内如何访问宿主机服务？

**A:**
- Linux: 使用 `host.docker.internal`
- 或使用宿主机的IP地址
- 或使用 `--network host` 模式

### Q4: 如何查看当前服务器IP？

**A:**
```bash
# 公网IP
curl ifconfig.me
curl ipinfo.io/ip

# 内网IP
hostname -I
ip addr show
```

### Q5: WebSocket连接失败？

**A:** 确保：
1. WebSocket URL使用正确的协议（ws:// 或 wss://）
2. 如果使用HTTPS，WebSocket也要用WSS
3. 代理服务器配置支持WebSocket升级

Nginx配置示例：
```nginx
location /ws/ {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### Q6: 健康检查失败？

**A:** 检查：
1. 应用是否正常启动
2. 端口是否正确
3. 健康检查路径是否正确 (`/health`)
4. 防火墙是否阻止

---

## 📞 技术支持

如遇到问题：
1. 查看日志: `docker-compose logs -f api`
2. 检查健康状态: `curl http://your-ip:8000/health`
3. 提交Issue: https://github.com/your-username/network-security/issues

---

**更新时间:** 2025-11-23
