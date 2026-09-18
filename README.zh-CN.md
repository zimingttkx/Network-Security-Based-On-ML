# NIPS — 网络入侵防御系统

[English](README.md) · **简体中文**

一个运行在服务器侧的 IPS：在 Linux 上拦截流量，先用规则引擎再用异常检测器对每个数据包打分，最后通过 iptables 丢弃恶意包。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
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

LUCID（基于 CNN 的 DDoS 检测器）是**可选**的。它默认不接入流水线，需要 TensorFlow（`pip install -e ".[lucid]"` 或 `pip install tensorflow`）和训练好的模型，并显式启用。见 `networksecurity/engine/lucid/`。

### 算法

- **Kitsune (NDSS'18)** — AfterImage 增量统计（100 维特征）+ KitNET 自编码器集成。在线训练，无需标签。
- **LUCID (IEEE TNSM 2020)** — 在 10 包流窗口（每包 11 维特征）上跑的 1D CNN。默认关闭，需要训练好的模型。

> **关于协议过滤：** 规则引擎的协议白名单只包含 TCP(6) 和 UDP(17)。其他任何协议——包括 **ICMP(1)**——默认都会被拦截。也就是说，合法的 ICMP（ping、PMTUD、traceroute）同样会被丢弃，除非其源地址在白名单中。如果你运行的网络依赖 ICMP，请把相关源地址加入白名单，或在启用实时拦截前先收紧该策略。

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

# 可选：LUCID CNN 检测器需要 TensorFlow，默认安装不包含它
#（未安装时 LUCID 适配器保持未激活状态）。
pip install -e ".[lucid]"     # 或：pip install tensorflow

# 可选：离线 pcap 测试（cli.py test --pcap）需要 scapy
pip install scapy
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
python cli.py test --pcap sample.pcap  # 离线检测测试（无需 root）
```

---

## API 参考

| 方法 | 端点 | 描述 |
| ------ | -------- | ----------- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/status` | 引擎状态、检测器、已封禁 IP（含内核级封禁）、检测循环健康度 |
| `GET` | `/api/v1/stats/overview` | 流量与阻断统计 |
| `GET` | `/api/v1/alerts` | 最近告警日志（分页） |
| `GET` | `/api/v1/rules` | 当前黑名单和白名单 |
| `GET` | `/api/v1/blocks` | 封禁升级状态（观察中 / 临时封禁 / 永久封禁） |
| `POST` | `/api/v1/rules/blacklist` | 将 IP 加入黑名单 |
| `DELETE` | `/api/v1/rules/blacklist/{ip}` | 从黑名单移除 IP |
| `POST` | `/api/v1/rules/whitelist` | 将 IP/CIDR 加入白名单 |
| `DELETE` | `/api/v1/rules/whitelist/{ip}` | 从白名单移除 IP |
| `POST` | `/api/v1/engine/start` | 启动实时拦截（Linux，需 root） |
| `POST` | `/api/v1/engine/stop` | 停止拦截并清理 iptables 规则 |

完整的交互式文档见 `/docs`。

**认证：**在 `config.yaml` 中设置 `api.auth_token`（或环境变量 `NIPS_API_TOKEN`）后，所有 `/api/v1/*` 调用都必须携带请求头 `X-API-Token: <token>`。留空表示关闭认证（仅限开发环境，服务启动时会打 WARNING）。`/health` 保持开放（用于存活探测）。

---

## 目录结构

```
app.py                         # FastAPI 应用入口
cli.py                         # CLI 管理工具
config/
  config.yaml                  # 引擎/拦截配置
networksecurity/
  engine/                      # 检测引擎
    detector.py                # BaseDetector 接口 + PacketInfo
    verdict.py                 # Verdict、Action、ThreatLevel 类型
    pipeline.py                # DetectionPipeline（多阶段链）
    rule_engine.py             # IP 黑名单/白名单、限速
    block_policy.py            # BLOCK 判决升级：strike 累计 → 临时封禁 → 永久封禁
    kitsune/                   # Kitsune 异常检测器（NDSS'18）
      afterimage.py            # 100 维增量统计
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
    dataset_loader.py          # NSL-KDD、CICIDS2017、UNSW-NB15（CSV / Parquet）
    pcap_loader.py             # PCAP 文件读取器
  utils/                       # 共享工具
    config.py                  # config.yaml 读取（engine / api / blocking 块）
scripts/                       # 基准测试、评估与回归检查
  benchmark.py                 # 吞吐量 + 规则引擎准确率
  benchmark_nslkdd.py          # NSL-KDD 检测基准
  attack_simulation.py         # 大规模攻击模拟
  build_unsw_pcap.py           # 用内置 UNSW-NB15 流记录重建真实流量 pcap
  evaluate_pcap.py             # 端到端 pcap 评估（按攻击类别报告）
  verify_*.py                  # 模块回归检查，含 CI 的 FPR 守卫
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
- 通过升级策略（`config.yaml` 的 `blocking:`）执行 BLOCK 判决，且**仅对 ML 检测器的 BLOCK 生效**。规则引擎的判决（黑名单命中、限速、协议过滤）是确定性的、已经逐包内联执行，因此不计 strike、不参与升级——这同时保证了操作员的黑名单条目永远不会被封禁生命周期改动。单次 ML BLOCK 只内联丢弃当前包，并给源 IP 计一次 strike。滚动窗口内累计达到 `strikes_threshold` 触发**临时封禁**——内核 DROP 加规则引擎黑名单*镜像*（带 TTL，到期自动解除；解除时只删除镜像，绝不触碰操作员自己的条目）；反复触发临时封禁会升级为**永久封禁**，写入 `rules.json`，下次启动时加载回规则引擎、在用户态逐包拦截——内核 DROP 本身**不会**被重新安装
- 关闭时清除自己添加的所有 iptables 规则

**仅支持 IPv4。** 只有 IPv4 的 TCP/UDP 流量会被重定向到 NFQUEUE 并被解析。IPv6 入站流量既不检测也不阻断——它会完全绕过本 IPS。在双栈（dual-stack）主机上，请另行防护 IPv6（例如用 `ip6tables` 设置策略）或直接禁用它。

`Interceptor` 从 `config.yaml` 读取 `safe_ips` 和 `nfqueue_num`；配置文件缺失或无法解析时回退到安全默认值（包含回环防护），不会在无保护状态下启动。

检测超时只会内联丢弃当前这个包（fail-closed），绝不提交永久封禁，因此检测慢不会误封合法 IP。

---

## 训练数据集准备

`DatasetLoader`（`networksecurity/data/dataset_loader.py`）把 NSL-KDD、CICIDS2017、UNSW-NB15 作为**带标签的 CSV 或 Parquet**加载，用于 LUCID/Kitsune 的监督训练。它要求每个文件**已经是带有标准列名的 CSV**（含表头；`.parquet` 后缀的文件按 Parquet 读取）——它**不会**自动识别或转换表头，也不处理原始的、无表头的 NSL-KDD `.txt` 发行版。在调用 `DatasetLoader` 之前，由用户自己负责把文件整理好。

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

### 用真实流量做离线测试

有两条路径可以在**不需要** root 和 iptables 的情况下验证检测流水线的行为——适合在真实抓包上确认效果：

- **真实 pcap（验证真实性能的首选）：** 抓包后离线跑过流水线。
  ```bash
  # 抓取 30 秒实时流量（抓包本身需要 root）
  sudo python -c "from scapy.all import sniff, wrpcap; wrpcap('cap.pcap', sniff(iface='en0', timeout=30))"
  # 离线检测——无需 root
  python cli.py test --pcap cap.pcap
  ```
  这样能暴露**真实**的误报率（例如合法 ICMP 被协议过滤拦截），下面的合成模拟做不到这一点。注意 Kitsune 大约需要 55k 个正常包才会离开训练模式，所以短抓包主要测的是规则引擎。
- **合成攻击模拟：** `scripts/attack_simulation.py` 生成带标签的流量并按攻击类别报告检出率。它的 ICMP/SSH 结果反映的是硬性协议规则和可分离的生成器分布，不是生产环境的准确率——快速模式下整体约 20% 的攻击检出率应视为下限，而非准确率声明。

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
