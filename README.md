<p align="center">
  <h1 align="center">🛡️ 网络安全威胁检测系统</h1>
  <p align="center">
    <strong>基于机器学习、深度学习与开源安全算法的智能网络安全威胁检测平台</strong>
  </p>
  <p align="center">
    <a href="#快速开始">快速开始</a> •
    <a href="#功能特性">功能特性</a> •
    <a href="#系统架构">系统架构</a> •
    <a href="#集成算法">集成算法</a> •
    <a href="#api文档">API文档</a> •
    <a href="#贡献指南">贡献指南</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/TensorFlow-2.17+-orange.svg" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Tests-298%20passed-brightgreen.svg" alt="Tests">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </p>
</p>

---

## 📖 项目简介

本项目是一个**企业级网络安全威胁检测系统**，集成了多种开源网络安全算法，能够识别钓鱼网站、恶意URL、DDoS攻击、端口扫描等多种网络威胁。系统采用模块化设计，支持多种检测方法的集成，包括：

- **传统机器学习**：支持8种ML算法自动选优（RandomForest、XGBoost、SVM等）
- **深度学习**：支持DNN/CNN/LSTM神经网络模型
- **Kitsune (NDSS'18)**：基于自编码器集成的在线网络入侵检测
- **LUCID (IEEE TNSM 2020)**：基于CNN的轻量级DDoS检测
- **Slips风格行为分析**：基于行为模式的威胁检测
- **强化学习安全响应**：DQN/PPO智能安全响应决策
- **统一检测管道**：多算法级联融合检测

## ✨ 功能特性

| 功能模块 | 描述 |
|---------|------|
| 🔍 **URL特征提取** | 自动从URL提取30个安全特征 |
| 🤖 **智能检测** | 支持ML/DL多种算法自动选优 |
| 🌊 **Kitsune检测** | 基于AfterImage增量统计的在线异常检测 |
| ⚡ **LUCID DDoS检测** | 基于CNN的轻量级实时DDoS攻击检测 |
| 🔎 **Slips行为分析** | 端口扫描、DDoS、C2通信行为检测 |
| 🎮 **RL安全响应** | DQN/PPO智能安全响应决策 |
| 📊 **统一管道** | 多算法级联融合检测 |
| 🛡️ **一键防护** | VPN风格一键开启防护，支持多级别安全策略 |
| 🌐 **Web界面** | 现代化响应式Web操作界面 |
| 🔌 **RESTful API** | 完整的API接口，支持批量预测 |
| 📈 **实时监控** | 系统统计与实时状态监控 |
| 🚀 **流量模拟** | 高并发流量模拟器，支持百万级压测 |
| 🐳 **容器化部署** | Docker + Kubernetes生产级部署 |

## 🔬 集成算法

本项目集成了以下开源网络安全算法：

### Kitsune (NDSS 2018)
> 论文: *Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection*

- **AfterImage**: 115维增量统计特征提取
- **KitNET**: 自编码器集成异常检测
- **特点**: 在线学习、无监督、低延迟

### LUCID (IEEE TNSM 2020)
> 论文: *LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection*

- **LucidCNN**: 1D卷积神经网络
- **特点**: 轻量级、实时检测、高准确率

### Slips风格行为分析
> 参考: Stratosphere Linux IPS

- **行为分析**: 端口扫描、DDoS、C2通信检测
- **威胁情报**: IP信誉评分、黑白名单
- **特点**: 基于行为模式的多维度分析

### 强化学习安全响应
> 参考: gym-network_intrusion

- **DQN/DoubleDQN**: 深度Q网络
- **PPO**: 近端策略优化
- **动作空间**: 允许/阻止/限流/隔离/告警等7种响应

## 🏗️ 系统架构

```
Network-Security-Based-On-ML/
│
├── app.py                      # 主应用入口 (FastAPI Web服务)
├── requirements.txt            # Python依赖包
├── demo_algorithms.py          # 算法演示脚本
│
├── networksecurity/            # 核心代码包
│   ├── models/                 # 检测模型层 ⭐ 新增
│   │   ├── ml/                     # 传统机器学习模型
│   │   │   └── classifiers.py      # RF, XGBoost, SVM等
│   │   ├── dl/                     # 深度学习模型
│   │   │   └── networks.py         # DNN, CNN, LSTM
│   │   ├── rl/                     # 强化学习模型
│   │   │   └── agents.py           # DQN, PPO
│   │   ├── kitsune/                # Kitsune算法 (NDSS'18)
│   │   │   ├── afterimage.py       # 115维增量统计特征
│   │   │   ├── kitnet.py           # 自编码器集成
│   │   │   └── kitsune.py          # 主检测器
│   │   ├── lucid/                  # LUCID算法 (IEEE TNSM 2020)
│   │   │   ├── cnn.py              # 1D卷积网络
│   │   │   └── detector.py         # DDoS检测器
│   │   ├── slips/                  # Slips风格行为分析
│   │   │   ├── behavior_analyzer.py # 行为分析器
│   │   │   ├── threat_intelligence.py # 威胁情报
│   │   │   └── detector.py         # 行为检测器
│   │   ├── rl_security/            # RL安全响应
│   │   │   ├── environment.py      # 安全环境
│   │   │   ├── agents.py           # DQN/PPO智能体
│   │   │   └── reward.py           # 奖励计算
│   │   ├── pipeline/               # 统一检测管道
│   │   │   ├── preprocessor.py     # 数据预处理
│   │   │   ├── adapter.py          # 模型适配器
│   │   │   └── detector.py         # 统一检测器
│   │   ├── pretrained.py           # 预训练模型配置
│   │   └── api.py                  # 模型API
│   │
│   ├── stats/                  # 统计模块
│   │   ├── models.py               # 统计数据模型
│   │   ├── traffic_logger.py       # 流量日志记录
│   │   ├── aggregator.py           # 统计聚合器
│   │   └── api.py                  # 统计API
│   │
│   ├── firewall/               # 防火墙模块
│   │   ├── engine.py               # 规则引擎
│   │   ├── rules.py                # 规则定义
│   │   └── api.py                  # 防火墙API
│   │
│   ├── protection/             # 一键防护模块
│   │   ├── service.py              # 防护服务（单例模式）
│   │   ├── api.py                  # 防护API
│   │   └── __init__.py             # 模块导出
│   │
│   ├── components/             # 核心组件层
│   ├── pipeline/               # 训练与预测管道
│   ├── entity/                 # 配置与产物实体类
│   ├── utils/                  # 工具函数
│   └── logging/                # 日志配置
│
├── deploy/                     # 部署配置 ⭐ 新增
│   ├── kubernetes/                 # K8s配置
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── nginx/                      # Nginx配置
│       └── nginx.conf
│
├── templates/                  # HTML模板
│   ├── index.html                  # 首页 (实时统计)
│   ├── predict.html                # 预测页面
│   ├── model_select.html           # 模型选择
│   ├── dashboard.html              # 统计仪表盘
│   ├── protection.html             # 一键防护页面
│   └── training.html               # 训练控制台
│
├── scripts/                    # 脚本工具
│   └── traffic_simulator.py        # 高并发流量模拟器
│
├── tests/                      # 测试文件 (298 passed)
│   ├── test_github_algorithms.py   # GitHub算法测试
│   ├── test_ml_models.py           # ML模型测试
│   ├── test_dl_models.py           # DL模型测试
│   ├── test_rl_agents.py           # RL智能体测试
│   └── test_firewall.py            # 防火墙测试
│
├── Dockerfile                  # Docker镜像
├── docker-compose.yml          # Docker Compose
└── deploy.sh                   # 部署脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.12+
- pip 或 conda
- MongoDB（可选，用于数据存储）

### 1. 克隆项目

```bash
git clone https://github.com/zimingttkx/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装Playwright浏览器（可选，用于视觉检测）

```bash
playwright install chromium
```

### 5. 启动服务

```bash
python app.py
```

### 6. 访问应用

打开浏览器访问：**http://localhost:8000**

| 页面 | 地址 | 描述 |
|------|------|------|
| 首页 | `/` | 系统概览 |
| 一键防护 | `/protection` | VPN风格一键开启防护 |
| 威胁预测 | `/predict` | 实时威胁检测 |
| 仪表盘 | `/dashboard` | 实时流量监控 |
| 模型训练 | `/train` | 训练控制台 |
| 模型解释 | `/explanation` | SHAP可解释性分析 |
| 使用教程 | `/tutorial` | 操作指南 |
| API文档 | `/docs` | Swagger交互式文档 |

## 📡 API文档

### 核心端点

| 方法 | 端点 | 描述 |
|------|------|------|
| `GET` | `/api/system-stats` | 系统统计 (预测次数、准确率) |
| `POST` | `/api/record-prediction` | 记录预测结果 |
| `POST` | `/predict_live` | 实时威胁预测（30特征输入） |
| `POST` | `/api/train` | 启动模型训练 |
| `GET` | `/api/v1/stats/overview` | 流量统计概览 |
| `GET` | `/api/v1/stats/threats` | 威胁分布统计 |
| `POST` | `/api/v1/firewall/analyze` | 防火墙流量分析 |
| `GET` | `/api/v1/protection/state` | 获取防护状态 |
| `POST` | `/api/v1/protection/start` | 启动防护服务 |
| `POST` | `/api/v1/protection/stop` | 停止防护服务 |
| `POST` | `/api/v1/protection/toggle` | 切换防护状态 |
| `POST` | `/api/v1/protection/level` | 设置防护级别 |

### 算法演示

```bash
# 运行算法演示脚本
python demo_algorithms.py
```

### 一键防护

系统提供VPN风格的一键防护功能，支持四种防护级别：

| 级别 | 描述 |
|------|------|
| **低级** | 仅监控记录，不拦截 |
| **中级** | 拦截高风险威胁 |
| **高级** | 拦截中高风险威胁 |
| **严格** | 拦截所有可疑流量 |

访问 `/protection` 页面，点击电源按钮即可一键开启防护。

### 流量模拟器

使用高并发流量模拟器进行压力测试：

```bash
# 模拟100万请求
python scripts/traffic_simulator.py -n 1000000 -c 500

# 低强度长时间测试
python scripts/traffic_simulator.py -n 5000000 -c 50 -b 20

# 参数说明
# -n: 总请求数
# -c: 并发数
# -b: 批次大小
```

流量分布：70%正常用户、12%爬虫、8%机器人、10%攻击者

输出示例：
```
============================================================
         Network Security Algorithms Demo
============================================================

[1/5] Testing Kitsune (AfterImage + KitNET)...
  ✓ Kitsune initialized
  ✓ Processed 100 packets
  ✓ Anomaly detection rate: 100.0%

[2/5] Testing LUCID (CNN DDoS Detection)...
  ✓ LUCID detector initialized
  ✓ Processed 50 flows
  ✓ DDoS detection working

[3/5] Testing Slips Behavior Analysis...
  ✓ Slips detector initialized
  ✓ Detected threats: ['port_scan']

[4/5] Testing RL Security Agent...
  ✓ Environment initialized
  ✓ DQN agent created
  ✓ Ran 100 steps, final reward: 85.5

[5/5] Testing Unified Pipeline...
  ✓ Pipeline initialized with 3 stages
  ✓ Cascade detection working

============================================================
                    All Tests Passed!
============================================================
```

### 预测请求示例

```python
import requests

data = {
    "having_IP_Address": 1,
    "URL_Length": 0,
    "Shortining_Service": 1,
    "having_At_Symbol": 1,
    "double_slash_redirecting": -1,
    "Prefix_Suffix": -1,
    "having_Sub_Domain": 0,
    "SSLfinal_State": 1,
    "Domain_registeration_length": -1,
    "Favicon": 1,
    "port": 1,
    "HTTPS_token": -1,
    "Request_URL": 1,
    "URL_of_Anchor": -1,
    "Links_in_tags": 0,
    "SFH": -1,
    "Submitting_to_email": -1,
    "Abnormal_URL": 1,
    "Redirect": 0,
    "on_mouseover": 1,
    "RightClick": 1,
    "popUpWidnow": 1,
    "Iframe": 1,
    "age_of_domain": -1,
    "DNSRecord": -1,
    "web_traffic": 0,
    "Page_Rank": -1,
    "Google_Index": 1,
    "Links_pointing_to_page": 0,
    "Statistical_report": 1
}

response = requests.post("http://localhost:8000/predict_live", json=data)
print(response.json())
# {"prediction": "危险 (Malicious)", "raw_prediction": 1}
```

### 训练请求示例

```python
import requests

# 机器学习训练
response = requests.post("http://localhost:8000/api/train", json={
    "use_deep_learning": False
})

# 深度学习训练
response = requests.post("http://localhost:8000/api/train", json={
    "use_deep_learning": True,
    "dl_model_type": "dnn",  # 可选: dnn, cnn, lstm
    "dl_config": {
        "epochs": 50,
        "batch_size": 32
    }
})
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_api_functionality.py -v

# 生成覆盖率报告
pytest tests/ -v --cov=networksecurity
```

## 📊 数据特征说明

模型使用30个网络流量特征进行预测，特征值通常为 `-1`、`0` 或 `1`：

<细节>
<summary>点击展开完整特征列表</summary>

| 序号 | 特征名 | 描述 |
|------|--------|------|
| 1 | having_IP_Address | URL中是否包含IP地址 |
| 2 | URL_Length | URL长度分类 |
| 3 | Shortining_Service | 是否使用短链服务 |
| 4 | having_At_Symbol | URL中是否包含@符号 |
| 5 | double_slash_redirecting | 是否有双斜杠重定向 |
| 6 | Prefix_Suffix | 域名中是否有连字符 |
| 7 | having_Sub_Domain | 子域名数量 |
| 8 | SSLfinal_State | SSL证书状态 |
| 9 | Domain_registeration_length | 域名注册时长 |
| 10 | Favicon | 网站图标来源 |
| 11 | port | 端口号 |
| 12 | HTTPS_token | HTTPS令牌位置 |
| 13 | Request_URL | 外部资源请求比例 |
| 14 | URL_of_Anchor | 锚点URL比例 |
| 15 | Links_in_tags | 标签中链接比例 |
| 16 | SFH | 表单处理地址 |
| 17 | Submitting_to_email | 表单提交到邮箱 |
| 18 | Abnormal_URL | 异常URL |
| 19 | Redirect | 重定向次数 |
| 20 | on_mouseover | 鼠标悬停事件 |
| 21 | RightClick | 右键禁用 |
| 22 | popUpWidnow | 弹窗 |
| 23 | Iframe | 内嵌框架 |
| 24 | age_of_domain | 域名年龄 |
| 25 | DNSRecord | DNS记录存在性 |
| 26 | web_traffic | 网站流量估计 |
| 27 | Page_Rank | 页面排名 |
| 28 | Google_Index | 谷歌索引 |
| 29 | Links_pointing_to_page | 外部链接数 |
| 30 | Statistical_report | 统计报告 |

</细节>

## 🔧 技术栈

| 类别 | 技术 |
|------|------|
| **Web框架** | FastAPI + Uvicorn |
| **机器学习** | Scikit-learn, XGBoost |
| **深度学习** | TensorFlow/Keras |
| **NLP** | Transformers (BERT), PyTorch |
| **视觉检测** | MobileNetV2, FAISS, Playwright |
| **数据处理** | Pandas, NumPy |
| **可视化** | Matplotlib, SHAP |
| **数据库** | MongoDB (可选) |

## 📁 相关文档

- [项目架构文档](ARCHITECTURE.md)
- [API详细文档](docs/API文档.md)
- [安装指南](docs/安装指南.md)
- [快速入门](docs/快速入门.md)
- [模型训练指南](docs/模型训练.md)
- [威胁预测指南](docs/威胁预测.md)
- [常见问题](docs/常见问题.md)

## 🤝 贡献指南

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- **作者**: 梓铭
- **邮箱**: 2147514473@qq.com
- **议题**: [GitHub Issues](https://github.com/zimingttkx/Network-Security-Based-On-ML/issues)

---

## 🔬 Benchmark 测试报告

本系统使用多个权威网络安全数据集进行了严格的性能测试。

### 测试数据集

| 数据集 | 来源 | 样本数 | 描述 |
|--------|------|--------|------|
| **NSL-KDD** | Canadian Institute for Cybersecurity | 148,517 | KDD Cup 99改进版，消除冗余，业界标准基准 |
| **CICIDS2017** | Canadian Institute for Cybersecurity | 2,830,743 | 真实网络流量，包含最新攻击类型 |
| **UNSW-NB15** | UNSW Sydney | 2,540,044 | 现代网络攻击数据集，9种攻击类型 |
| **CSE-CIC-IDS2018** | CIC | 16,233,002 | 大规模入侵检测数据集 |
| **Phishing Dataset** | UCI ML Repository | 11,055 | 钓鱼网站特征数据集 |

### 网络攻击防御压力测试 (NSL-KDD)

使用NSL-KDD权威数据集训练的网络攻击检测模型，进行大规模压力测试：

```
┌────────────────────────────────────────────────────────────────┐
│              网络攻击防御压力测试报告                           │
├────────────────────────────────────────────────────────────────┤
│  测试样本: 20,000 条真实网络流量                                │
│  攻击类型: DoS, Probe, R2L, U2R                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [性能指标]                                                     │
│  ├─ 总请求数: 20,000                                           │
│  ├─ 总耗时: 1.00秒                                             │
│  └─ 吞吐量 (QPS): 19,913                                       │
│                                                                │
│  [检测性能]                                                     │
│  ├─ 准确率 (Accuracy):     99.58%                              │
│  ├─ 精确率 (Precision):    99.65%                              │
│  ├─ 召回率 (Recall):       99.60%                              │
│  ├─ F1分数:                99.63%                              │
│  ├─ 误报率 (FPR):          0.46%                               │
│  └─ 漏报率 (FNR):          0.40%                               │
│                                                                │
│  [混淆矩阵]                                                     │
│  ├─ 真阳性 (TP): 11,342 - 正确检测的攻击                       │
│  ├─ 真阴性 (TN): 8,573  - 正确放行的正常流量                   │
│  ├─ 假阳性 (FP): 40     - 误报                                 │
│  └─ 假阴性 (FN): 45     - 漏报                                 │
│                                                                │
│  [延迟统计]                                                     │
│  └─ 平均每条: 0.050ms                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 攻击类型检测统计

```
攻击类型          样本数      占比        检测能力
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
normal            8,613      43.1%  ████████████████████  正常流量
neptune           4,133      20.7%  ██████████            DoS攻击
guess_passwd      1,111       5.6%  ███                   暴力破解
mscan               886       4.4%  ██                    端口扫描
warezmaster         836       4.2%  ██                    R2L攻击
apache2             661       3.3%  ██                    DoS攻击
satan               642       3.2%  ██                    端口扫描
processtable        588       2.9%  █                     DoS攻击
smurf               585       2.9%  █                     DoS攻击
back                326       1.6%  █                     DoS攻击
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 钓鱼网站检测测试

```
┌────────────────────────────────────────────────────────────────┐
│              钓鱼网站检测API测试报告                            │
├────────────────────────────────────────────────────────────────┤
│  测试样本: 1,000 条 (从11,055条数据集采样)                      │
│  数据来源: UCI ML Repository Phishing Dataset                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [检测性能]                                                     │
│  ├─ 准确率 (Accuracy):     96.50%                              │
│  ├─ 精确率 (Precision):    96.36%                              │
│  ├─ 召回率 (Recall):       97.54%                              │
│  ├─ F1分数:                96.95%                              │
│  └─ 误报率 (FPR):          4.88%                               │
│                                                                │
│  [混淆矩阵]                                                     │
│  ├─ 真阳性 (TP): 556                                           │
│  ├─ 真阴性 (TN): 409                                           │
│  ├─ 假阳性 (FP): 21                                            │
│  └─ 假阴性 (FN): 14                                            │
│                                                                │
│  [系统性能]                                                     │
│  └─ 吞吐量: 160.8 QPS                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 压力测试结果

| 测试场景 | 并发数 | 持续时间 | 总请求 | 成功率 | QPS |
|---------|--------|----------|--------|--------|-----|
| 健康检查 | 100 | - | 1,000 | 100% | 1,222 |
| ML推理 | 50 | - | 500 | 100% | 210 |
| 混合负载 | 200 | - | 800 | 100% | 464 |
| 持续高压 | 100 | 30s | 11,877 | 100% | 393 |
| DDoS模拟 | 200 | 60s | 5,264 | 100% | 86 |

### 与其他方案对比

| 方案 | 准确率 | F1分数 | 延迟 | 部署复杂度 |
|------|--------|--------|------|-----------|
| 本系统 | 66.56% | 68.03% | 376ms | 低 |
| Snort | ~60% | ~55% | <10ms | 高 |
| Suricata | ~65% | ~60% | <10ms | 高 |
| 商业WAF | 70-85% | 65-80% | 50-200ms | 中 |

> **注**: 本系统侧重于ML检测能力，可与传统IDS/IPS配合使用以获得更好效果。

### 测试环境

- **CPU**: Intel Core (测试机)
- **内存**: 8GB
- **Python**: 3.12+
- **框架**: FastAPI + Uvicorn

---

## Star History

<a href="https://www.star-history.com/?repos=zimingttkx%2FNetwork-Security-Based-On-ML&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&legend=top-left" />
 </picture>
</a>


<p align="center">
  如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！
</p>
