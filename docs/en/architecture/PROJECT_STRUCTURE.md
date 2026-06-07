# 项目结构说明

```
Network-Security-Based-On-ML/
│
├── app.py                      # 主应用入口 (FastAPI Web服务)
├── requirements.txt            # Python依赖包
├── setup.py                    # 包安装配置
├── Dockerfile                  # Docker镜像构建
├── docker-compose.yml          # Docker Compose配置
├── deploy.sh                   # 部署脚本
│
├── networksecurity/            # 核心代码包
│   ├── __init__.py
│   ├── api/                    # API模块
│   │   └── app.py
│   ├── components/             # 核心组件
│   │   ├── data_ingestion.py       # 数据摄入
│   │   ├── data_validation.py      # 数据验证
│   │   ├── data_transformation.py  # 数据转换
│   │   ├── model_trainer.py        # 模型训练
│   │   ├── dl_model_trainer.py     # 深度学习训练
│   │   ├── ensemble_detector.py    # 集成检测器
│   │   ├── bert_classifier.py      # BERT分类器
│   │   └── visual_similarity.py    # 视觉相似度
│   ├── config/                 # 配置管理
│   │   └── config_manager.py
│   ├── constant/               # 常量定义
│   │   └── training_pipeline/
│   ├── entity/                 # 实体类
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   ├── exception/              # 异常处理
│   │   └── exception.py
│   ├── firewall/               # 防火墙模块
│   │   ├── api.py                  # 防火墙API
│   │   ├── detector.py             # 威胁检测器
│   │   └── captcha.py              # 验证码
│   ├── logging/                # 日志模块
│   │   └── logger.py
│   ├── models/                 # 检测模型
│   │   ├── api.py                  # 模型API
│   │   ├── pretrained.py           # 预训练模型
│   │   ├── ml/                     # 机器学习模型
│   │   │   ├── base.py
│   │   │   ├── classifiers.py
│   │   │   └── anomaly.py
│   │   ├── dl/                     # 深度学习模型
│   │   │   ├── base.py
│   │   │   ├── classifiers.py
│   │   │   └── autoencoder.py
│   │   ├── rl/                     # 强化学习模型
│   │   │   ├── base.py
│   │   │   ├── agents.py
│   │   │   └── environment.py
│   │   ├── kitsune/                # Kitsune算法 (NDSS'18)
│   │   │   ├── afterimage.py
│   │   │   ├── kitnet.py
│   │   │   ├── kitsune.py
│   │   │   └── feature_extractor.py
│   │   ├── lucid/                  # LUCID算法 (IEEE TNSM 2020)
│   │   │   ├── cnn.py
│   │   │   ├── detector.py
│   │   │   └── dataset_parser.py
│   │   ├── slips/                  # Slips行为分析
│   │   │   ├── behavior_analyzer.py
│   │   │   ├── flow_analyzer.py
│   │   │   ├── threat_intelligence.py
│   │   │   └── detector.py
│   │   ├── rl_security/            # RL安全响应
│   │   │   ├── agents.py
│   │   │   ├── environment.py
│   │   │   └── reward.py
│   │   └── pipeline/               # 统一检测管道
│   │       ├── preprocessor.py
│   │       ├── adapter.py
│   │       └── detector.py
│   ├── pipeline/               # 训练管道
│   │   ├── training_pipeline.py
│   │   └── batch_prediction.py
│   ├── stats/                  # 统计模块
│   │   ├── api.py
│   │   ├── models.py
│   │   ├── traffic_logger.py
│   │   ├── aggregator.py
│   │   └── demo_data.py
│   └── utils/                  # 工具函数
│       ├── url_feature_extractor.py
│       ├── web_content_extractor.py
│       ├── main_utils/
│       │   └── utils.py
│       └── ml_utils/
│           ├── data_validator.py
│           ├── model_explanation.py
│           ├── model/
│           │   ├── estimator.py
│           │   ├── ensemble.py
│           │   └── automl.py
│           └── metric/
│               └── classification_metric.py
│
├── benchmarks/                 # 性能测试脚本
│   ├── algorithm_benchmark.py      # 算法对比测试
│   ├── network_attack_stress_test.py # 网络攻击压力测试
│   ├── stress_test.py              # API压力测试
│   ├── attack_simulation.py        # 攻击模拟
│   ├── nslkdd_test.py              # NSL-KDD测试
│   └── real_attack_test.py         # 真实攻击测试
│
├── scripts/                    # 工具脚本
│   ├── demo_algorithms.py          # 算法演示
│   └── network_attack_model.py     # 网络攻击模型训练
│
├── tests/                      # 单元测试
│   ├── conftest.py
│   ├── test_ml_models.py
│   ├── test_dl_models.py
│   ├── test_rl_agents.py
│   ├── test_firewall.py
│   ├── test_github_algorithms.py
│   ├── test_stats_*.py
│   └── ...
│
├── templates/                  # HTML模板
│   ├── index.html                  # 首页
│   ├── predict.html                # 预测页面
│   ├── training.html               # 训练控制台
│   ├── dashboard.html              # 统计仪表盘
│   ├── model_select.html           # 模型选择
│   ├── explanation.html            # 模型解释
│   └── tutorial.html               # 使用教程
│
├── static/                     # 静态资源
│   └── css/
│
├── deploy/                     # 部署配置
│   ├── k8s/                        # Kubernetes配置
│   │   ├── deployment.yaml
│   │   └── config.yaml
│   └── nginx/                      # Nginx配置 (如有)
│
├── config/                     # 应用配置
│   └── config.yaml
│
├── data/                       # 数据文件
│   ├── phisingData.csv             # 钓鱼数据集
│   └── nsl_kdd_test.csv            # NSL-KDD测试集
│
├── data_schema/                # 数据模式
│   └── schema.yaml
│
├── models/                     # 训练好的模型
│   ├── model.pkl                   # 主模型
│   ├── preprocessor.pkl            # 预处理器
│   ├── network_attack_model.pkl    # 网络攻击模型
│   └── network_attack_scaler.pkl   # 网络攻击标准化器
│
├── artifacts/                  # 训练产物
│   └── ...
│
├── logs/                       # 日志文件
│
├── docs/                       # 文档
│   ├── API_REFERENCE.md            # API参考
│   ├── DEPLOYMENT_GUIDE.md         # 部署指南
│   └── ...
│
└── .github/                    # GitHub配置
    ├── workflows/                  # CI/CD工作流
    │   ├── ci.yml
    │   ├── test.yml
    │   └── lint.yml
    └── ISSUE_TEMPLATE/             # Issue模板
```

## 命名规范

- **Python文件**: 小写下划线命名 (snake_case)
- **类名**: 大驼峰命名 (PascalCase)
- **函数/变量**: 小写下划线命名 (snake_case)
- **常量**: 大写下划线命名 (UPPER_SNAKE_CASE)
- **目录**: 小写下划线命名 (snake_case)

## 模块说明

| 模块 | 说明 |
|------|------|
| `networksecurity/` | 核心业务逻辑 |
| `benchmarks/` | 性能测试和压力测试脚本 |
| `scripts/` | 工具脚本和演示程序 |
| `tests/` | 单元测试和集成测试 |
| `deploy/` | 部署相关配置 |
| `docs/` | 项目文档 |
