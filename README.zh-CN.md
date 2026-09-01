# NIPS — 网络入侵防御系统

[English](README.md) · **简体中文**

一个运行在服务器侧的 IPS：在 Linux 上拦截流量，先用规则引擎再用异常检测器对每个数据包打分，最后通过 iptables 丢弃恶意包。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.17+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 提交代码前请先阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [CONTRIBUTING.md](CONTRIBUTING.md)。CI 会拒绝 `networksecurity/` 中的模拟/桩代码。

---

## 工作原理

```
流入流量
      |
      v
[规则引擎 Rule Engine] ------> 阻断（黑名单、限速、协议过滤）
      | 通过
      v
[Kitsune] ----------> 阻断（AfterImage + KitNET 异常检测）
      | 通过
      v
[放行 ALLOW]
```

规则引擎确定性地处理已知恶意流量（黑名单、白名单、限速、协议白名单）。通过的数据包交给 Kitsune——一个无监督的包级异常检测器，先在正常流量上训练，再用重建误差（RMSE）偏离程度来标记异常。

LUCID（基于 CNN 的 DDoS 检测器）是**可选**的。它默认不接入流水线，需要训练好的 TensorFlow 模型并显式启用。见 `networksecurity/engine/lucid/`。

### 算法

- **Kitsune (NDSS'18)** — AfterImage 增量统计（115 维特征）+ KitNET 自编码器集成。在线训练，无需标签。
- **LUCID (IEEE TNSM 2020)** — 在 10 包流窗口（每包 11 维特征）上跑的 1D CNN。默认关闭，需要训练好的模型。

---

## 快速开始

### 环境要求

- Python 3.12+
- 实时拦截需要 Linux（nfqueue + iptables，需 root）
- macOS / 其他平台可用于开发与离线 pcap 测试

### 1. 克隆仓库

```bash
git clone https://github.com/zimingttkx/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML
```

### 2. 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 运行 API

```bash
python app.py
# API 文档见 http://localhost:8000/docs
```

### 4. CLI

```bash
python cli.py start                  # 启动实时拦截（Linux，需 root）
python cli.py stop                   # 停止实时拦截（通过 API）
python cli.py status                 # 引擎状态
python cli.py block 1.2.3.4          # 封禁某个 IP
python cli.py unblock 1.2.3.4        # 解封某个 IP
python cli.py whitelist 10.0.0.0/8   # 将某个子网加入白名单
python cli.py rules                  # 列出黑名单/白名单条目
python cli.py alerts --last 20       # 查看最近告警（通过 API）
python cli.py test --pcap sample.pcap  # 离线检测测试
```

---

## API 参考

| 方法 | 端点 | 描述 |
| ------ | -------- | ----------- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/status` | 引擎状态、检测器、已封禁 IP |
| `GET` | `/api/v1/stats/overview` | 流量与阻断统计 |
| `GET` | `/api/v1/alerts` | 最近告警日志（分页） |
| `GET` | `/api/v1/rules` | 当前黑名单和白名单 |
| `POST` | `/api/v1/rules/blacklist` | 将 IP 加入黑名单 |
| `DELETE` | `/api/v1/rules/blacklist/{ip}` | 从黑名单移除 IP |
| `POST` | `/api/v1/rules/whitelist` | 将 IP/CIDR 加入白名单 |
| `DELETE` | `/api/v1/rules/whitelist/{ip}` | 从白名单移除 IP |
| `POST` | `/api/v1/engine/start` | 启动实时拦截（Linux，需 root） |
| `POST` | `/api/v1/engine/stop` | 停止拦截并清理 iptables 规则 |

完整的交互式文档见 `/docs`。

---

## 目录结构

```
app.py                         # FastAPI 应用入口
cli.py                         # CLI 管理工具
config/
  config.yaml                  # 引擎/拦截配置
templates/                     # Web 状态页模板
networksecurity/
  engine/                      # 检测引擎
    detector.py                # BaseDetector 接口 + PacketInfo
    verdict.py                 # Verdict、Action、ThreatLevel 类型
    pipeline.py                # DetectionPipeline（多阶段链）
    rule_engine.py             # IP 黑名单/白名单、限速
    kitsune/                   # Kitsune 异常检测器（NDSS'18）
      afterimage.py            # 115 维增量统计
      kitnet.py                # 自编码器集成
      kitsune.py               # 编排器
      detector_adapter.py      # BaseDetector 适配器
    lucid/                     # LUCID DDoS 检测器（IEEE TNSM 2020，可选）
      cnn.py                   # 1D CNN 模型
      dataset_parser.py        # 流缓冲与特征提取
      detector.py              # 编排器
      detector_adapter.py      # BaseDetector 适配器
  interception/                # Linux 流量拦截
    nfqueue_handler.py         # NFQUEUE 绑定与数据包捕获
    packet_parser.py           # 原始 IPv4 数据包解析器
    iptables.py                # iptables 规则管理
    interceptor.py             # 实时拦截器（nfqueue + pipeline）
  features/                    # 特征提取
    flow_extractor.py          # 逐流统计特征
    feature_registry.py        # 特征集注册表
  data/                        # 数据加载
    dataset_loader.py          # NSL-KDD、CICIDS2017、UNSW-NB15
    pcap_loader.py             # PCAP 文件读取器
scripts/                       # 基准测试与评估
  benchmark.py                 # 吞吐量 + 规则引擎准确率
  benchmark_nslkdd.py          # NSL-KDD 检测基准
  attack_simulation.py         # 大规模攻击模拟
```

---

## 实时拦截（仅 Linux）

```bash
# 1. 安装 nfqueue 库
pip install NetfilterQueue

# 2. 以 root 权限运行
sudo python -c "
from networksecurity.interception import Interceptor
from networksecurity.engine import DetectionPipeline
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector

pipeline = DetectionPipeline()
pipeline.add_detector(KitsuneDetector())

interceptor = Interceptor(pipeline)
interceptor.start()  # 阻塞运行。Ctrl+C 停止。
"
```

拦截器会：

- 写入 iptables 规则，把流量重定向到 NFQUEUE
- 回环流量完全不进检测流水线——`lo` 接口到达的包在 NFQUEUE 规则之前就被 ACCEPT；回环源地址（`127.0.0.0/8`、`::1`）永远不会被永久封禁（本机流量不可能是攻击者；封掉 DNS stub `127.0.0.53` 会静默瘫痪本机域名解析）
- 不动 SSH（22 端口）
- 每一次 ML/规则引擎的 BLOCK 判决都会同步写入规则引擎黑名单，封禁因此能跨重启保留（`rules.json`），下次启动时自动重新应用到内核
- 关闭时清除自己添加的所有 iptables 规则

`Interceptor` 从 `config.yaml` 读取 `safe_ips` 和 `nfqueue_num`；配置文件缺失时这些配置会被静默忽略——请保留并提交 `config.yaml`。

检测超时只会内联丢弃当前这个包（fail-closed），绝不提交永久封禁，因此检测慢不会误封合法 IP。

---

## 训练数据集准备

`DatasetLoader`（`networksecurity/data/dataset_loader.py`）把 NSL-KDD、CICIDS2017、UNSW-NB15 作为**带标签的 CSV**加载，用于 LUCID/Kitsune 的监督训练。它要求每个文件**已经是带有标准列名的 CSV**（含表头）——它**不会**自动识别或转换表头，也不处理原始的、无表头的 NSL-KDD `.txt` 发行版。在调用 `DatasetLoader` 之前，由用户自己负责把文件整理好。

各数据集要求的格式：

| 数据集 | 要求 | 说明 |
| --- | --- | --- |
| **NSL-KDD** | 带表头 CSV，共 43 列：41 个标准 NSL-KDD 特征，然后是 `difficulty`，最后是 `label` | 官方的 `KDDTrain+.txt` / `KDDTest+.txt` **没有表头**——加载前需补上 41 个标准特征名 + `difficulty` + `label`。二元标签：`normal`/`normal.` → 0（正常），其余 → 1（攻击）。 |
| **UNSW-NB15** | 带表头 CSV，含二元 `label` 列（0/1），以及元数据列 `id`、`attack_cat` | `attack_cat` 会被自动丢弃（否则会泄漏标签）。 |
| **CICIDS2017** | 带表头 CSV，含 `Label` 列（大写 L），以及 `Flow ID` / `Timestamp` / `Source IP` / `Destination IP` | 这 4 个元数据列会被自动丢弃。`BENIGN` → 0，其余 → 1。 |

类别型列会做 one-hot 编码（`get_dummies`，`drop_first`），缺失值填 0，结果以 `float32` 返回。若需要训练/测试编码对齐，请用 `train_test_split()`——它会在训练集上拟合编码，再把测试集 reindex 到相同列。

---

## 基准测试

两个脚本用于在你自己的机器上跑出数据——下面的数字未在各环境验证，实际结果会有差异：

- `scripts/benchmark.py` — 用合成的普通流量训练 Kitsune，再报告规则引擎准确率、训练/检测吞吐量和攻击检出率。
- `scripts/benchmark_nslkdd.py` — 下载 NSL-KDD，把流记录映射成合成数据包，用普通流训练 Kitsune，报告精确率/召回率/误报率。

为什么在 NSL-KDD 上检出率偏低：NSL-KDD 记录是**流级摘要**，不是真实抓包。把每条流映射成几个包，会丢掉 Kitsune 依赖的时序和突发模式。大流量型攻击（DoS、probe）比内容型攻击（R2L、U2R）更能保留映射后的特征——后者在包级看起来和正常 TCP 没有区别。把各攻击类别的数字当作这一局限性的说明，而不是实测准确率。

规则引擎本身是精确的：黑名单/白名单、协议过滤、限速都是确定性的，且始终在 ML 阶段之前执行。

---

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 分层设计、数据流、模块边界、红线
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR 工作流、提交前检查清单、我们不接受的内容
- [CODE_STYLE.md](CODE_STYLE.md) — 编码规范、导入规则、系统调用校验
- [SECURITY.md](SECURITY.md) — 漏洞报告、部署最佳实践
- [CHANGELOG.md](CHANGELOG.md) — 发布历史
- API 参考：`http://localhost:8000/docs`（Swagger）

---

## 联系方式

- **作者**：梓铭
- **邮箱**：2147514473@qq.com
- **Issues**：[GitHub Issues](https://github.com/zimingttkx/Network-Security-Based-On-ML/issues)

## 许可证

MIT — 详见 [LICENSE](LICENSE)
