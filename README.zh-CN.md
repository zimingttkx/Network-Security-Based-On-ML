# NIPS — 网络入侵防御系统

[English](README.md) · **简体中文**

使用机器学习进行实时网络入侵检测与防御。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.17+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> **贡献者**：提交代码前请先阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [CONTRIBUTING.md](CONTRIBUTING.md)。所有 PR 都会通过 CI 检查模拟代码。

---

## 概述

NIPS 是一个服务器端的网络入侵防御系统。它拦截流入流量，从网络流中提取统计特征，并通过多阶段检测流水线将每个数据包分类为良性或恶意。恶意流量通过 iptables 在内核层面进行阻断。

### 检测流水线

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
[LUCID] ------------> 阻断（基于 CNN 的 DDoS 流检测）
      | 通过
      v
[放行 ALLOW]
```

### 集成的算法

- **Kitsune (NDSS'18)** — AfterImage 增量统计（115 个特征）+ KitNET 自编码器集成，用于在线异常检测。无监督、低延迟。
- **LUCID (IEEE TNSM 2020)** — 轻量级 1D CNN，用于实时 DDoS 检测。每个流窗口 10 个数据包，每个数据包 11 个特征。

---

## 快速开始

### 环境要求

- Python 3.12+
- Linux（用于通过 nfqueue/iptables 进行实时拦截）
- macOS（用于开发和离线测试）

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
python cli.py start            # 启动实时拦截（Linux，需 root）
python cli.py stop             # 停止实时拦截（通过 API）
python cli.py status           # 引擎状态
python cli.py block 1.2.3.4    # 封禁某个 IP
python cli.py unblock 1.2.3.4  # 解封某个 IP
python cli.py whitelist 10.0.0.0/8  # 将某个子网加入白名单
python cli.py rules            # 列出黑名单/白名单条目
python cli.py alerts --last 20 # 查看最近告警（通过 API）
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

## 架构

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
    lucid/                     # LUCID DDoS 检测器（IEEE TNSM 2020）
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

拦截器会自动：
- 设置 iptables 规则，将流量重定向到 NFQUEUE
- 保护 SSH（22 端口）和回环接口
- 在关闭时清理所有 iptables 规则

---

---

## 基准测试

在 NSL-KDD（UNSW 和加拿大网络安全研究所提供的标准 NIDS 数据集，125,973 条训练流，11,849 条测试流）上进行基准测试。将流映射为逐包 `PacketInfo` 对象，并通过 Kitsune 流水线（AfterImage 115 维特征 + KitNET 自编码器集成）进行处理。

测试环境：GitHub Codespaces（2 vCPU，8 GB RAM）。

### Kitsune — 无监督异常检测

| 指标 | 值 |
| ------ | ----- |
| 训练数据包 | 150,000 |
| 训练吞吐量 | 819 pkt/s |
| 检测吞吐量 | 1,126 pkt/s |
| 持续吞吐量 | 1,085 pkt/s |
| 精确率 | 89.0% |
| 误报率 | 3.2% |

### 各类攻击检出率

| 攻击类别 | 检出率 | 说明 |
| --------------- | -------------- | ----- |
| DoS（SYN 洪泛、Neptune、Smurf） | 15% | 大流量型——包级突发模式可部分检出 |
| Probe（端口扫描、IP 扫描） | 9% | 低速型——将流映射为包会丢失扫描节奏 |
| R2L（口令猜测、warezclient） | <1% | 内容型——在包级与正常 TCP 难以区分 |
| U2R（缓冲区溢出、rootkit） | <1% | 内容型——AfterImage 看到的是正常大小、正常标志位的包 |

### 解读

Kitsune 是一种**无监督包级**检测器。89% 的精确率意味着当它标记某个东西时，几乎可以肯定是恶意的。3.2% 的误报率意味着正常流量很少被误分类——对于处于阻断模式的 NIPS 来说是可以接受的。

较低的召回率（尤其是 R2L 和 U2R）反映了该基准测试的一个根本局限：NSL-KDD 记录是**流级摘要**，而不是真实的数据包捕获。R2L/U2R 攻击在逐包层面看起来与正常流量完全相同。DoS 和 Probe 攻击更有前景，因为它们的流量型模式在「流→包」映射后仍然存在。在真实 pcap 流量上的逐包检测准确率对于 DoS 和 Probe 类别预计会显著更高。

### 规则引擎 — 确定性过滤

规则引擎为已知 IP（黑名单/白名单）、协议过滤和限速提供微秒级、100% 准确的过滤。与 Kitsune 异常检测相结合，提供了纵深防御：先进行快速的基于规则的预过滤，再进行基于 ML 的异常检测以应对未知威胁。

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

---

## Star 历史

<a href="https://www.star-history.com/?repos=zimingttkx%2FNetwork-Security-Based-On-ML&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&legend=top-left" />
 </picture>
</a>
