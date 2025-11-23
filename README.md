# 🔐 Network Security Threat Detection System

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 企业级网络安全威胁检测系统 - 基于机器学习的实时网络流量异常检测与预警平台

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [模型训练](#模型训练)
- [部署指南](#部署指南)
- [监控与运维](#监控与运维)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🎯 项目概述

Network Security Threat Detection System 是一个工业级的网络安全威胁检测平台，利用先进的机器学习算法对网络流量进行实时分析，识别潜在的恶意活动和安全威胁。

### 核心功能

- 🤖 **智能威胁检测**：使用集成学习算法（XGBoost、LightGBM、CatBoost等）进行高精度威胁识别
- 📊 **实时分析**：毫秒级响应的实时网络流量分析
- 🔄 **自动化训练**：AutoML自动超参数优化，持续提升模型性能
- 📈 **可视化监控**：Prometheus + Grafana 实时监控系统运行状态
- 🚀 **高可用部署**：支持Docker、Kubernetes等多种部署方式
- 🔐 **企业级安全**：完整的认证、授权、加密机制

## ✨ 核心特性

### 🎓 机器学习能力

- **多模型集成**：支持10+种机器学习算法
  - RandomForest, GradientBoosting, AdaBoost
  - XGBoost, LightGBM, CatBoost
  - SVM, KNN, LogisticRegression, GaussianNB

- **AutoML优化**：基于Optuna的自动超参数调优
- **集成学习**：Voting、Stacking、Blending策略
- **不平衡数据处理**：SMOTE、RandomUnderSampling等技术
- **特征工程**：自动特征选择和转换
- **模型版本管理**：MLflow完整的实验追踪

### 🏗️ 工程能力

- **RESTful API**：基于FastAPI的高性能异步API
- **配置管理**：YAML配置文件 + Pydantic验证
- **日志系统**：结构化日志 + 自动轮转
- **异常处理**：完善的异常捕获和错误追踪
- **测试覆盖**：单元测试 + 集成测试 + 性能测试
- **CI/CD**：GitHub Actions自动化测试和部署
- **容器化**：Docker多阶段构建优化
- **编排管理**：Kubernetes生产级部署配置

### 📊 监控能力

- **性能监控**：Prometheus指标采集
- **可视化**：Grafana仪表板
- **告警系统**：实时告警通知
- **日志聚合**：ELK Stack集成（可选）
- **分布式追踪**：Jaeger集成（可选）

## 🏛️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│                   (Web UI / Mobile App / API Client)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     API Gateway (Nginx)                      │
│              (Load Balancing / Rate Limiting)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  FastAPI Application Layer                   │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │   Training   │  Prediction  │  Monitoring  │            │
│  │   Service    │   Service    │   Service    │            │
│  └──────┬───────┴──────┬───────┴──────┬───────┘            │
└─────────┼──────────────┼──────────────┼────────────────────┘
          │              │              │
┌─────────▼─────┐ ┌──────▼──────┐ ┌────▼──────────────┐
│  ML Pipeline  │ │Model Serving│ │ Metrics Collector │
│  ┌─────────┐  │ │  ┌───────┐  │ │  (Prometheus)     │
│  │Ingestion│  │ │  │ Model │  │ └───────────────────┘
│  │Validation│ │ │  │ Cache │  │
│  │Transform│  │ │  └───────┘  │
│  │Training │  │ └─────────────┘
│  │Evaluation│ │
│  └─────────┘  │
└────────┬──────┘
         │
┌────────▼──────────────────┬──────────────────┐
│  Data Layer               │  Storage Layer    │
│  ┌──────────┐            │  ┌────────────┐  │
│  │ MongoDB  │            │  │  Models    │  │
│  │ (Primary)│            │  │  Storage   │  │
│  └──────────┘            │  └────────────┘  │
│  ┌──────────┐            │  ┌────────────┐  │
│  │  Redis   │            │  │   Logs     │  │
│  │ (Cache)  │            │  │  Storage   │  │
│  └──────────┘            │  └────────────┘  │
└───────────────────────────┴──────────────────┘
```

## 🚀 快速开始

### 前置要求

- Python 3.12+
- MongoDB 7.0+
- Docker & Docker Compose (可选)
- Kubernetes (生产环境部署)

### 本地开发环境设置

#### 1. 克隆项目

```bash
git clone https://github.com/your-username/network-security.git
cd network-security
```

#### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量

创建 `.env` 文件：

```bash
# MongoDB配置
MONGO_DB_URL=mongodb+srv://username:password@cluster.mongodb.net/

# 应用配置
APP_ENV=development
APP_DEBUG=true
LOG_LEVEL=INFO

# MLflow配置
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

#### 5. 运行数据库迁移（如需要）

```bash
# 推送数据到MongoDB
python push_data.py
```

#### 6. 启动应用

```bash
# 方式1: 直接运行
python -m networksecurity.api.app

# 方式2: 使用uvicorn
uvicorn networksecurity.api.app:app --reload --host 0.0.0.0 --port 8000
```

访问（根据实际部署环境替换域名/IP）：
- API文档: http://127.0.0.1:8000/api/docs 或 http://your-server-ip:8000/api/docs
- 健康检查: http://127.0.0.1:8000/health 或 http://your-server-ip:8000/health
- Metrics: http://127.0.0.1:8000/metrics 或 http://your-server-ip:8000/metrics

### 使用Docker快速启动

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

## 📁 项目结构

```
network-security/
├── networksecurity/              # 主应用包
│   ├── api/                      # API层
│   │   └── app.py               # FastAPI应用
│   ├── components/               # 核心组件
│   │   ├── data_ingestion.py   # 数据摄取
│   │   ├── data_validation.py  # 数据验证
│   │   ├── data_transformation.py  # 数据转换
│   │   └── model_trainer.py    # 模型训练
│   ├── config/                   # 配置管理
│   │   └── config_manager.py   # 配置加载器
│   ├── pipeline/                 # 数据管道
│   │   ├── training_pipeline.py  # 训练管道
│   │   └── batch_prediction.py   # 批量预测
│   ├── utils/                    # 工具函数
│   │   ├── ml_utils/            # ML工具
│   │   │   ├── model/           # 模型相关
│   │   │   │   ├── automl.py   # AutoML
│   │   │   │   ├── ensemble.py # 集成学习
│   │   │   │   └── estimator.py  # 模型估计器
│   │   │   └── metric/          # 评估指标
│   │   └── main_utils/          # 通用工具
│   ├── exception/                # 异常处理
│   ├── logging/                  # 日志系统
│   └── constant/                 # 常量定义
├── config/                       # 配置文件
│   └── config.yaml              # 主配置
├── deployment/                   # 部署配置
│   ├── kubernetes/              # K8s配置
│   ├── prometheus/              # 监控配置
│   └── nginx/                   # Nginx配置
├── tests/                        # 测试文件
│   ├── test_data_ingestion.py
│   └── test_config.py
├── .github/                      # GitHub配置
│   └── workflows/               # CI/CD工作流
│       ├── ci.yml
│       └── deploy.yml
├── Dockerfile                    # Docker配置
├── docker-compose.yml           # Docker Compose配置
├── requirements.txt             # Python依赖
├── setup.py                     # 包安装配置
└── README.md                    # 本文件
```

## ⚙️ 配置说明

### 配置文件结构

主配置文件位于 `config/config.yaml`，包含以下部分：

```yaml
app:                    # 应用配置
database:               # 数据库配置
data_pipeline:          # 数据管道配置
model_training:         # 模型训练配置
  models:              # 支持的模型
    xgboost:
      enabled: true
      params: {...}
  hyperparameter_tuning:  # 超参数优化
  ensemble:            # 集成学习
api:                    # API配置
logging:                # 日志配置
monitoring:             # 监控配置
security:               # 安全配置
deployment:             # 部署配置
```

### 环境变量优先级

环境变量 > config.yaml > 默认值

## 📖 API文档

### 核心端点

#### 1. 健康检查

```http
GET /health
```

响应：
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": 3600.5
}
```

#### 2. 训练模型

```http
POST /api/v1/train
```

响应：
```json
{
  "status": "success",
  "message": "训练完成",
  "metrics": {
    "train_f1": 0.95,
    "test_f1": 0.93
  }
}
```

#### 3. 预测（JSON）

```http
POST /api/v1/predict
Content-Type: application/json

{
  "data": [[1.0, 2.0, 3.0, 4.0]]
}
```

响应：
```json
{
  "predictions": [1],
  "probabilities": [0.85],
  "threat_level": ["危险 (Malicious)"]
}
```

#### 4. 预测（文件上传）

```http
POST /api/v1/predict/file
Content-Type: multipart/form-data

file: data.csv
```

详细API文档：http://your-server-ip:8000/api/docs

## 🎓 模型训练

### 训练流程

1. **数据摄取**：从MongoDB读取原始数据
2. **数据验证**：检查数据质量和完整性
3. **数据转换**：特征工程和数据预处理
4. **模型训练**：训练多个模型并选择最优
5. **模型评估**：在测试集上评估性能
6. **模型保存**：保存最优模型

### 运行训练

```bash
# 方式1: 通过API（替换为实际IP/域名）
curl -X POST http://your-server-ip:8000/api/v1/train

# 方式2: 命令行
python main.py

# 方式3: 使用AutoML
python -c "from networksecurity.utils.ml_utils.model.automl import AutoMLOptimizer; ..."
```

### AutoML使用示例

```python
from networksecurity.utils.ml_utils.model.automl import AutoMLOptimizer

# 创建优化器
optimizer = AutoMLOptimizer(n_trials=100, timeout=3600)

# 优化XGBoost
best_params, best_score = optimizer.optimize('xgb', X_train, y_train)
print(f"最佳参数: {best_params}")
print(f"最佳得分: {best_score}")
```

### 集成学习示例

```python
from networksecurity.utils.ml_utils.model.ensemble import EnsembleBuilder

# 创建集成构建器
ensemble_builder = EnsembleBuilder()

# 创建投票集成
estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cb', cb_model)
]
voting_model = ensemble_builder.create_voting_ensemble(
    estimators, voting='soft'
)

# 训练并评估
voting_model.fit(X_train, y_train)
```

## 🚢 部署指南

### Docker部署

#### 单容器部署

```bash
# 构建镜像
docker build -t network-security-api:latest .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -e MONGO_DB_URL="your_mongo_url" \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/final_models:/app/final_models \
  --name network-security-api \
  network-security-api:latest
```

#### Docker Compose部署

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### Kubernetes部署

#### 1. 创建命名空间

```bash
kubectl create namespace production
```

#### 2. 创建Secrets

```bash
# 编辑secrets配置
cp deployment/kubernetes/secrets.yaml.example deployment/kubernetes/secrets.yaml
vim deployment/kubernetes/secrets.yaml

# 应用配置
kubectl apply -f deployment/kubernetes/secrets.yaml
```

#### 3. 部署应用

```bash
# 应用所有配置
kubectl apply -f deployment/kubernetes/

# 查看部署状态
kubectl get pods -n production
kubectl get svc -n production

# 查看日志
kubectl logs -f deployment/network-security-api -n production
```

#### 4. 配置自动扩缩容

HPA已自动配置，基于CPU和内存使用率自动扩展Pod数量（3-10个）

```bash
# 查看HPA状态
kubectl get hpa -n production
```

### 生产环境清单

- [ ] 配置HTTPS证书
- [ ] 设置MongoDB副本集
- [ ] 配置备份策略
- [ ] 设置监控告警
- [ ] 配置日志聚合
- [ ] 性能测试和调优
- [ ] 灾难恢复计划
- [ ] 安全审计

## 📊 监控与运维

### Prometheus指标

访问 http://your-server-ip:9090 查看Prometheus控制台

核心指标：
- `api_requests_total` - API请求总数
- `api_request_latency_seconds` - API请求延迟
- `predictions_total` - 预测总数
- `training_jobs_total` - 训练任务总数

### Grafana仪表板

访问 http://your-server-ip:3000 (默认用户名/密码: admin/admin)

预配置仪表板：
- API性能监控
- 模型预测统计
- 系统资源使用
- 告警历史

### 日志查看

```bash
# Docker日志
docker-compose logs -f api

# Kubernetes日志
kubectl logs -f deployment/network-security-api -n production

# 本地日志
tail -f logs/networksecurity_*.log
```

## 👨‍💻 开发指南

### 代码规范

项目使用以下工具保证代码质量：

```bash
# 代码格式化
black networksecurity/

# 代码检查
flake8 networksecurity/

# 类型检查
mypy networksecurity/
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行测试并生成覆盖率报告
pytest tests/ --cov=networksecurity --cov-report=html

# 运行特定测试
pytest tests/test_data_ingestion.py -v
```

### 添加新模型

1. 在 `config/config.yaml` 中添加模型配置
2. 在 `networksecurity/components/model_trainer.py` 中添加模型初始化代码
3. 添加相应的单元测试
4. 更新文档

### 提交代码

```bash
# 1. 创建特性分支
git checkout -b feature/your-feature-name

# 2. 提交代码
git add .
git commit -m "feat: add your feature"

# 3. 推送到远程
git push origin feature/your-feature-name

# 4. 创建Pull Request
```

提交信息规范（Conventional Commits）：
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `style:` 代码格式
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具变动

## ❓ 常见问题

### Q: 如何更新模型？

A: 通过调用 `/api/v1/train` 端点触发重新训练，训练完成后模型会自动更新。

### Q: 支持哪些数据格式？

A: 目前支持CSV和JSON格式，数据需包含特定的特征列。

### Q: 如何配置告警？

A: 编辑 `deployment/prometheus/alerts/api_alerts.yml` 配置告警规则。

### Q: 性能优化建议？

A:
1. 启用Redis缓存
2. 增加API workers数量
3. 使用模型量化
4. 启用GZIP压缩
5. 配置CDN

### Q: 如何备份数据？

A: MongoDB数据备份：
```bash
mongodump --uri="your_mongo_url" --out=/backup/dir
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码审查流程

1. 所有PR需要至少1个审查者批准
2. CI测试必须通过
3. 代码覆盖率不能降低
4. 需要更新相关文档

## 📜 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- 作者：梓铭
- Email: 2147514473@qq.com
- 项目地址: https://github.com/your-username/network-security

## 🙏 致谢

感谢以下开源项目：

- [FastAPI](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [MLflow](https://mlflow.org/)
- [Prometheus](https://prometheus.io/)

---

⭐ 如果这个项目对你有帮助，请给一个Star！