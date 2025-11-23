# ⚡ 快速启动指南

## 🎯 一分钟启动项目

### 步骤1: 克隆项目
```bash
git clone https://github.com/your-username/network-security.git
cd network-security
```

### 步骤2: 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，至少设置 MONGO_DB_URL
vim .env
```

### 步骤3: 启动（三选一）

#### 选项A: 使用Docker Compose（推荐）
```bash
docker-compose up -d
```
✅ 包含所有服务（API + MongoDB + Prometheus + Grafana + Redis + Nginx）

#### 选项B: 仅启动API容器
```bash
docker build -t network-security-api .
docker run -d -p 8000:8000 --env-file .env network-security-api
```
⚠️ 需要外部MongoDB

#### 选项C: 本地Python运行
```bash
pip install -r requirements.txt
python -m networksecurity.api.app
```
⚠️ 需要Python 3.12+和外部MongoDB

---

## 🌐 访问地址速查表

### 本地开发环境

| 服务 | 访问地址 | 端口 |
|-----|---------|------|
| **API文档** | http://127.0.0.1:8000/api/docs | 8000 |
| **健康检查** | http://127.0.0.1:8000/health | 8000 |
| **指标监控** | http://127.0.0.1:8000/metrics | 8000 |
| **Prometheus** | http://127.0.0.1:9090 | 9090 |
| **Grafana** | http://127.0.0.1:3000 | 3000 |
| **MLflow** | http://127.0.0.1:5000 | 5000 |
| **MongoDB** | mongodb://127.0.0.1:27017 | 27017 |
| **Redis** | redis://127.0.0.1:6379 | 6379 |

### 局域网访问

将 `127.0.0.1` 替换为你的本机IP（查看方法见下方）

示例: `http://192.168.1.100:8000/api/docs`

### 服务器部署

将 `127.0.0.1` 替换为服务器公网IP或域名

示例: `http://your-server-ip:8000/api/docs`

---

## 🔍 查看本机IP

### Mac/Linux
```bash
# 查看所有网络接口
ifconfig

# 快速查看主要IP（Mac）
ipconfig getifaddr en0

# 查看公网IP
curl ifconfig.me
```

### Windows
```bash
# 查看所有网络接口
ipconfig

# 查看公网IP
curl ifconfig.me
```

---

## 📝 常用命令

### Docker Compose管理
```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止所有服务
docker-compose down

# 重启服务
docker-compose restart api

# 查看服务状态
docker-compose ps
```

### 健康检查
```bash
# 检查API健康状态
curl http://127.0.0.1:8000/health

# 查看指标
curl http://127.0.0.1:8000/metrics
```

### 模型训练
```bash
# 通过API触发训练
curl -X POST http://127.0.0.1:8000/api/v1/train

# 本地运行训练脚本
python main.py
```

### 预测
```bash
# JSON预测
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0]]}'

# 文件预测
curl -X POST http://127.0.0.1:8000/api/v1/predict/file \
  -F "file=@data.csv"
```

---

## 🚨 故障排查

### API无法访问？

1. **检查服务是否启动**
   ```bash
   docker-compose ps
   # 或
   ps aux | grep uvicorn
   ```

2. **检查端口是否被占用**
   ```bash
   lsof -i :8000
   # 或
   netstat -an | grep 8000
   ```

3. **查看日志**
   ```bash
   docker-compose logs -f api
   # 或
   tail -f logs/networksecurity_*.log
   ```

4. **检查防火墙**
   ```bash
   # Ubuntu
   sudo ufw status

   # CentOS
   sudo firewall-cmd --list-all
   ```

### MongoDB连接失败？

1. **检查MongoDB是否运行**
   ```bash
   docker-compose ps mongodb
   ```

2. **测试连接**
   ```bash
   mongosh "your_mongodb_url"
   ```

3. **检查环境变量**
   ```bash
   cat .env | grep MONGO
   ```

### Docker容器无法启动？

1. **查看容器日志**
   ```bash
   docker logs network-security-api
   ```

2. **检查镜像是否构建成功**
   ```bash
   docker images | grep network-security
   ```

3. **重新构建**
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

---

## 🔐 安全提示

### 生产环境部署前必做：

- [ ] 修改 `.env` 中的默认密码
- [ ] 配置HTTPS/SSL证书
- [ ] 启用API认证
- [ ] 配置防火墙规则
- [ ] 限制MongoDB访问IP
- [ ] 设置Grafana管理员密码
- [ ] 定期备份数据

### 敏感信息：
⚠️ 永远不要将 `.env` 文件提交到Git
⚠️ 生产环境使用强密码
⚠️ 定期更新依赖包

---

## 📚 更多文档

- [完整README](README.md) - 详细的项目文档
- [部署指南](DEPLOYMENT_GUIDE.md) - 各种环境部署详解
- [优化总结](OPTIMIZATION_SUMMARY.md) - 项目优化记录
- [API文档](http://127.0.0.1:8000/api/docs) - 在线API文档

---

## 🆘 获取帮助

遇到问题？
1. 查看 [常见问题](#故障排查)
2. 阅读 [部署指南](DEPLOYMENT_GUIDE.md)
3. 提交 [Issue](https://github.com/your-username/network-security/issues)
4. 联系邮箱: 2147514473@qq.com

---

**提示:**
- 首次启动可能需要下载Docker镜像，请耐心等待
- 确保系统有足够的资源（至少2GB RAM）
- 推荐使用Docker Compose方式启动，包含所有必要服务

**快速测试:**
```bash
# 启动后执行
curl http://127.0.0.1:8000/health

# 应该返回:
# {"status":"healthy","version":"2.0.0","uptime":123.45}
```

✅ 看到上述返回说明部署成功！
