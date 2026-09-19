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

- **Kitsune (NDSS'18)** — AfterImage 增量统计（90 维特征）+ KitNET 自编码器集成。在线训练，无需标签。当链路层头部缺失（实时 NFQUEUE 场景）时，MAC 通道使用 `(protocol, ttl)` 代理键，避免方差退化为零。宽限期（`fm_grace_period`、`ad_grace_period`）允许在检测开始前先预热；此期间数据包只记录不拦截。
- **LUCID (IEEE TNSM 2020)** — 在 10 包流窗口（每包 11 维特征）上跑的 1D CNN。默认关闭，需要训练好的模型，且在配置中设置 `engine.lucid.model_path`；权重用 `scripts/train_lucid.py` 生成（见下文"训练 LUCID"）。

> **关于协议过滤：** 规则引擎的协议白名单只包含 TCP(6) 与 UDP(17)，凡是被它检查到的其他协议——包括 **ICMP(1)**——都会拦截。但在实时拦截中，只有 TCP 与 UDP 会被导入 NFQUEUE（`interception.intercept_icmp` 默认关闭），因此 ICMP 在那里**既不被检查、也不被拦截**：由主机自身的防火墙决定。把 `interception.intercept_icmp` 设为 true 才能让 ICMP 进入流水线，然后用 `engine.rule_engine.allowed_icmp_types` 按类型放行——整协议封禁会一并打断 Path MTU Discovery（type 3 "frag needed"），导致大连接被黑洞，所以有用的配置是"按类型放行"而不是一刀切封禁。离线 pcap 测试（`cli.py test --pcap`）确实会走到协议过滤，因为不论何种协议，包都会进入引擎。

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

### 3. 配置

`config/config.yaml` 同时驱动引擎和实时拦截：

- `engine.kitsune.*`：宽限期、阈值百分位、learning_rate（传给 AfterImage）
- `engine.lucid.model_path`：设置路径即启用 LUCID；空字符串表示禁用
- `api.auth_token`：设置后启用认证；空字符串表示关闭认证（仅开发环境）
- `api.host` / `api.port`：`python app.py` 的监听地址与端口
- `interception.safe_ips`：添加永远不会被封禁的 IP（回环默认包含）
- `interception.intercept_icmp` / `engine.rule_engine.allowed_icmp_types`：ICMP 策略（见上文协议过滤说明）
- `storage.*`：事件库路径、行数上限与保留窗口（见"告警、审计与指标"）
- `logging.*`：日志级别、轮转文件与 syslog 转发

### 4. 运行 API

```bash
python app.py
# /docs、/redoc 和 OpenAPI schema 在生产环境中全部关闭。
```

### 5. CLI

```bash
python cli.py start                  # 启动实时拦截（Linux，需 root）
python cli.py stop                   # 停止实时拦截（通过 API）
python cli.py status                 # 引擎状态
python cli.py block 1.2.3.4          # 封禁某个 IP（POST /api/v1/rules/blacklist）
python cli.py unblock 1.2.3.4        # 解封某个 IP（DELETE /api/v1/rules/blacklist/{ip}）
python cli.py whitelist --ip 10.0.0.0/8   # 将某个子网加入白名单（拒绝 /0 默认路由）
python cli.py unwhitelist --ip 10.0.0.0/8 # 从白名单移除
python cli.py rules                  # 列出黑名单/白名单条目
python cli.py reload                 # 把改过的 rules.json / 配置应用到运行中的引擎
python cli.py alerts --last 20       # 查看已存储告警（最新在前，走 API）
python cli.py alerts --source-ip 203.0.113.7 --action block
python cli.py alerts --since 2026-09-19T00:00:00 --format csv > alerts.csv
python cli.py audit --last 20        # 谁改了哪条规则，结果如何
python cli.py audit --result 401     # 被拒绝的管理请求
python cli.py test --pcap sample.pcap  # 离线检测测试（无需 root）
```

#### 配置示例

```yaml
interception:
  nfqueue_num: 0
  intercept_icmp: false     # 把 ICMP 也导入 NFQUEUE，按类型策略才会生效
  safe_ips:                 # 永远不会被封锁的 IP（回环受保护）
    - "127.0.0.1"
    - "::1"
engine:
  kitsune:
    fm_grace_period: 5000   # 特征映射训练包数
    ad_grace_period: 50000  # 异常检测器训练包数
    threshold_percentile: 99.0
  rule_engine:
    allowed_protocols: [6, 17]   # TCP、UDP；其余全部拦截
    allowed_icmp_types: []       # 协议 1 未列入时仍放行的 ICMP 类型，例如 [0, 3, 4, 8, 11] 可保住 PMTUD 与 ping
    rate_limit:
      window_seconds: 1.0
      max_connections_per_window: 100
blocking:                    # BLOCK 判决升级策略（见"实时拦截"）
  strikes_threshold: 5       # 窗口内累计 BLOCK 次数达到该值触发临时封禁
  strikes_window: 300.0
  temp_ban_seconds: 600.0
  temp_ban_count_to_perm: 3  # 完成的临时封禁次数达到该值触发永久封禁
api:
  auth_token: ""             # 空 = 关闭认证（仅开发）；NIPS_API_TOKEN 环境变量优先
  cors_origins:              # 显式白名单——不支持 "*"
    - "http://localhost:8000"
```

`engine/start` 时 API/CLI 从该文件读取 `interception`、`engine`、`blocking`、`api` 各块并在运行时应用。文件缺失或格式错误时，各加载器回退到安全默认值（包含回环保护），不会崩溃。

---

## API 参考

| 方法 | 端点 | 描述 |
| ------ | -------- | ----------- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/status` | 引擎状态、检测器、已封禁 IP（含内核级封禁）、检测循环健康度 |
| `GET` | `/api/v1/stats/overview` | 流量与阻断统计 |
| `GET` | `/api/v1/alerts` | 已存储告警：`limit`、`offset`、`source_ip`、`action`、`since`、`until`、`format=json\|csv\|jsonl` |
| `GET` | `/api/v1/audit` | 管理审计：`limit`、`offset`、`actor`、`result`、`since`、`until`、`format` |
| `GET` | `/api/v1/rules` | 当前黑名单和白名单 |
| `GET` | `/api/v1/blocks` | 封禁升级状态（观察中 / 临时封禁 / 永久封禁） |
| `POST` | `/api/v1/rules/blacklist` | 将 IP 加入黑名单 |
| `DELETE` | `/api/v1/rules/blacklist/{ip}` | 从黑名单移除 IP |
| `POST` | `/api/v1/rules/whitelist` | 将 IP/CIDR 加入白名单 |
| `DELETE` | `/api/v1/rules/whitelist/{ip}` | 从白名单移除 IP |
| `POST` | `/api/v1/rules/reload` | 重新读取 rules.json 与 config.yaml 中可热更的引擎参数 |
| `GET` | `/api/v1/signatures` | 已声明的签名规则及其命中计数 |
| `POST` | `/api/v1/signatures` | 新增或修改签名（重复 id 即替换该条） |
| `DELETE` | `/api/v1/signatures/{id}` | 删除签名 |
| `POST` | `/api/v1/engine/start` | 启动实时拦截（Linux，需 root） |
| `POST` | `/api/v1/engine/stop` | 停止拦截并清理 iptables 规则 |
| `GET` | `/metrics` | Prometheus 文本指标（与 `/api/v1/*` 同样需要 token） |

两个 `DELETE` 路由使用 `{ip:path}`，因此 CIDR 条目同样可删除（`/api/v1/rules/blacklist/10.0.0.0%2F8`，或直接写未编码形式）。

**认证：**在 `config.yaml` 中设置 `api.auth_token`（或环境变量 `NIPS_API_TOKEN`）后，所有 `/api/v1/*` 调用都必须携带请求头 `X-API-Token: <token>`。留空表示关闭认证（仅限开发环境，服务启动时会打 WARNING）。`/health` 保持开放（用于存活探测）。

### 告警、审计与指标

检测事件与每一次管理操作都会写入 SQLite（WAL）库 `storage.events_db`（默认 `data/events.db`）——重启不丢数据，`retention_days` 与 `max_rows` 控制文件规模。

- **告警**（`/api/v1/alerts`、`cli.py alerts`）——每次 BLOCK 判决一行，白名单变更也会记录。`format=csv|jsonl` 可导出给 SIEM；单次请求最多 1000 行。
- **审计**（`/api/v1/audit`、`cli.py audit`）——每条规则/引擎变更，以及每次被拒绝的尝试（401/422）各一行，含对端地址、方法、路径、目标与结果。API 只有一个共享 token，因此 `actor` 标识的是主机，不是具体用户。
- **指标**（`/metrics`）——处理/拦截包数、检测器状态、黑名单规模、封禁升级计数、事件库健康度。

检测路径不会等待磁盘：`record_alert` 只投递到有界缓冲区，由后台线程批量落盘。缓冲区溢出或批次失败时，`nips_alert_events_dropped_total` / `nips_event_store_write_errors_total` 计数上升，`/api/v1/status` 的 `event_store` 字段也会报告——审计链不完整是可见的，不会静默。数据库不可用时，读取回退到内存中最近 500 条事件，同时 `event_store.degraded` 为 true。

`logging.file` 增加轮转日志文件，`logging.syslog_address` 转发到 syslog（平台套接字，或 `host:port` UDP）；目标不可达时只告警并跳过，不阻塞启动。

### 签名规则

黑名单回答的是"这个源是不是坏的"，限速回答的是"有没有人发得太快"。两者都回答不了这条需求：**"当 203.0.113.0/24 对 TCP/22 超过每分钟 50 次会话时才丢弃"**——直接封网段会连带里面的合法用户，而全局限速无法按源和端口收窄。签名规则就是这个条件的合取：

```bash
# 先观察：只计数，不丢包
python cli.py signature add --id ssh-brute --src 203.0.113.0/24 \
    --protocol tcp --dport 22 --min-packets 50 --window 60 --action log
python cli.py signature list                    # 规则及其命中计数
python cli.py signature add --id ssh-brute --src 203.0.113.0/24 \
    --protocol tcp --dport 22 --min-packets 50 --window 60    # 再改为执行
python cli.py signature delete ssh-brute
```

| 字段 | 含义 |
|---|---|
| `src` / `dst` | IP 或 CIDR；`/0` 默认路由会被拒绝 |
| `protocol` | `tcp`、`udp`、`icmp` 或数字 |
| `dport` / `sport` | 0-65535。只给端口而不给协议时默认按 TCP——这些字节偏移在 ICMP 里是 echo 的 id/序号，在那里匹配端口没有意义 |
| `tcp_flags` | 对 6 位标志字段做精确匹配（`0x02` = SYN），仅对 TCP 有效 |
| `min_packets` + `window_seconds` | 同一源在窗口内命中 N 次后才触发 |
| `action` | `block` 内联丢弃该包；`log` 只计数并放行到 ML 阶段 |

- **没有任何匹配条件**的规则会被拒绝：配合 `action=block`，一次误调用就会丢弃全部流量。
- 按声明顺序求值、首个命中生效；位置在黑名单之后（被列名的源就按"黑名单命中"上报）、全局限速之前（收窄的规则不会被它掩盖）。
- 签名 BLOCK 与其他规则引擎判决一样**逐包内联执行**：不计 strike、不下发内核 DROP。对带次数条件的规则这是有意为之——一条持久内核规则会在触发它的条件消失后继续生效。要彻底封源，请用黑名单。
- 每源的命中计数受 LRU 上限约束（每条规则 1 万个源）：不设上限时，伪造源洪泛会让表每包增长一个条目，把检测功能变成内存耗尽漏洞。
- 签名持久化在 `rules.json` 的 `"signatures"` 中，并遵循上文的热加载规则：手工编辑最多 30 秒生效（或 `cli.py reload`），且一条非法条目会整份拒绝，而不是应用一半。
- API：`GET`/`POST /api/v1/signatures`、`DELETE /api/v1/signatures/{id}`；校验实现在引擎里，因此 API、文件与热加载拒绝的规格完全一致。新增与删除都会进审计。

### 热加载

`rules.json` 与 `config/config.yaml` 按 mtime 被监视，运行中的引擎最多 30 秒内拾取修改；`POST /api/v1/rules/reload`（或 `cli.py reload`）立即应用。无需重启——重启代价很高，因为 Kitsune 要从零重新训练。

- `rules.json` 按**替换**语义应用，删掉的条目会真正停止生效（启动时是合并语义，只会新增）。
- `engine.rule_engine.rate_limit.*` 与 `allowed_protocols` 在下一个包即生效。
- 文件损坏时整体拒绝：在线规则保持原样，`/api/v1/status` 的 `reload.failures` 上升，该次尝试以 `reload_failed` 记入审计。
- 内核本来就不会执行的条目（回环 / `safe_ips`）与启动时一样被清理，并在 `dropped_unenforceable` 中报告。
- Kitsune 的 `fm_grace_period`、`ad_grace_period`、`threshold_percentile`、`learning_rate` **不会**热应用——它们描述的是检测器如何训练，改动必须重启；重载摘要会列出这几项。

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
      afterimage.py            # 90 维增量统计
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
    packet_parser.py           # 原始 IPv4/IPv6 数据包解析器
    iptables.py                # iptables 规则管理
    interceptor.py             # 实时拦截器（nfqueue + pipeline）
  features/                    # 特征提取
    flow_extractor.py          # 逐流统计特征
    feature_registry.py        # 特征集注册表
  data/                        # 数据加载
    dataset_loader.py          # NSL-KDD、CICIDS2017、UNSW-NB15（CSV / Parquet）
    pcap_loader.py             # PCAP 文件读取器
  observability/               # 持久化事件与指标（只做存储/日志）
    alert_store.py             # SQLite 告警 + 审计，批量写入且不阻塞调用方
    metrics.py                 # Prometheus 文本指标
    log_setup.py               # 级别 / 轮转文件 / syslog 路由
  utils/                       # 共享工具
    config.py                  # config.yaml 读取（engine / api / blocking / storage / logging 块）
    validation.py              # IP/CIDR 校验与黑名单拒绝规则
scripts/                       # 基准测试、评估与回归检查
  benchmark.py                 # 吞吐量 + 规则引擎准确率
  benchmark_nslkdd.py          # NSL-KDD 检测基准
  attack_simulation.py         # 大规模攻击模拟
  build_unsw_pcap.py           # 用内置 UNSW-NB15 流记录重建真实流量 pcap
  train_lucid.py               # 训练 LUCID CNN 并产出 engine.lucid.model_path 指向的权重
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
- 除非开启 `interception.intercept_icmp`，只把 TCP 与 UDP 导入 NFQUEUE；开启后由 `allowed_icmp_types` 决定引擎接受哪些 ICMP 类型
- 通过升级策略（`config.yaml` 的 `blocking:`）执行 BLOCK 判决，且**仅对 ML 检测器的 BLOCK 生效**。规则引擎的判决（黑名单命中、限速、协议过滤）是确定性的、已经逐包内联执行，因此不计 strike、不参与升级——这同时保证了操作员的黑名单条目永远不会被封禁生命周期改动。单次 ML BLOCK 只内联丢弃当前包，并给源 IP 计一次 strike。滚动窗口内累计达到 `strikes_threshold` 触发**临时封禁**——内核 DROP 加规则引擎黑名单*镜像*（带 TTL，到期自动解除；解除时只删除镜像，绝不触碰操作员自己的条目）；反复触发临时封禁会升级为**永久封禁**，写入 `rules.json`，下次启动时加载回规则引擎、在用户态逐包拦截——内核 DROP 本身**不会**被重新安装
- 关闭时清除自己添加的所有 iptables 规则

**双栈，且退化时如实报告。** IPv4 与 IPv6 的 TCP/UDP 都会被重定向进 NFQUEUE、解析（含 IPv6 扩展头链）并通过 `ip6tables` 拦截。若 `ip6tables` 不可用，拦截器照常启动，但会**拒绝**所有 IPv6 封禁而不是假装成功，并把这个缺口报出来：`/api/v1/status` 的 `ipv6_intercepted: false`、`/metrics` 的 `nips_ipv6_intercepted 0`。双栈主机上请确认这个指标——被静默跳过的第二个地址族，正是出事之前没人会注意到的那种缺口。

三条解析限制依旧存在，全部 fail-closed 并计入 `nfqueue_parse_failed`：非首片 IP 分片不含传输头，因此丢弃而不是误读；AH/ESP 包在未认证的情况下无法走完扩展头链，因此丢弃而不是把密文当 TCP 头解析；IPv6 扩展头链超过 6 层视为构造包处理。
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

### 训练 LUCID

LUCID 是本项目里唯一有监督的检测器，所以必须先有权重文件，`engine.lucid.model_path` 才有东西可指：

```bash
# 1. 先看标签意味着什么——不需要 TensorFlow，也不写任何文件
python scripts/train_lucid.py --pcap capture.pcap \
    --attackers 203.0.113.0/24 --victims 10.0.0.1 --inspect

# 2. 训练并保存（先 pip install -e ".[lucid]"）
python scripts/train_lucid.py --pcap capture.pcap \
    --attackers attackers.txt --victims 10.0.0.1 --out models/lucid_cnn.h5
```

`--inspect` 会打印抓包能切出多少个完整窗口、攻击/正常的比例、以及有多少流在窗口填满前就过期了——这几个数字决定你写的地址到底有没有标上东西，省掉一次白跑的训练。要点：

- `--attackers`/`--victims` 可以写单个地址或 CIDR。非法条目直接报错而不是跳过：攻击列表里一个拼写错误，恰好会让模型最该学会的那部分流量失去标签。
- 按 LUCID 的约定，一个窗口里**多数**包的任意一端涉及 attacker 或 victim 地址即算攻击——所以一旦列了 victim，所有流向它的包都会被标成攻击。如果抓包里含该主机的正常业务，就只列 attacker。
- 训练复用在线特征路径，避免出现"按一种表示训练、按另一种表示打分"。
- 标签单侧、完整窗口为 0、或窗口少于 4 个都会被拒绝并打印原因，而不是产出一个只预测单一类别却看起来健康的模型。

---

## 基准测试

两个脚本用于在你自己的机器上跑出数据——下面的数字未在各环境验证，实际结果会有差异：

- `scripts/benchmark.py` — 用合成的普通流量训练 Kitsune，再报告规则引擎准确率、训练/检测吞吐量和攻击检出率。
- `scripts/benchmark_nslkdd.py` — 下载 NSL-KDD，把流记录映射成合成数据包，用普通流训练 Kitsune，报告精确率/召回率/误报率。

为什么在 NSL-KDD 上检出率偏低：NSL-KDD 记录是**流级摘要**，不是真实抓包。把每条流映射成几个包，会丢掉 Kitsune 依赖的时序和突发模式。大流量型攻击（DoS、probe）比内容型攻击（R2L、U2R）更能保留映射后的特征——后者在包级看起来和正常 TCP 没有区别。把各攻击类别的数字当作这一局限性的说明，而不是实测准确率。

规则引擎本身是精确的：黑名单/白名单、协议过滤、限速都是确定性的，且始终在 ML 阶段之前执行。限速只统计 TCP SYN（ACK 未置位）和 UDP 数据报；已建立的 TCP 会话（ACK/数据/FIN）不消耗限速额度。

### 用真实流量做离线测试

有两条路径可以在**不需要** root 和 iptables 的情况下验证检测流水线的行为——适合在真实抓包上确认效果：

- **真实 pcap（验证真实性能的首选）：** 抓包后离线跑过流水线。
  ```bash
  # 抓取 30 秒实时流量（抓包本身需要 root）
  sudo python -c "from scapy.all import sniff, wrpcap; wrpcap('cap.pcap', sniff(iface='en0', timeout=30))"
  # 离线检测——无需 root
  python cli.py test --pcap cap.pcap
  ```
  这样能暴露**真实**的误报率（例如合法 ICMP 被协议过滤拦截——离线路径中 ICMP 确实会进入引擎，而实时链路在 `intercept_icmp: false` 下不会）。注意 Kitsune 大约需要 55k 个正常包才会离开训练模式，所以短抓包主要测的是规则引擎。
- **合成攻击模拟：** `scripts/attack_simulation.py` 生成带标签的流量并按攻击类别报告检出率。它的 ICMP/SSH 结果反映的是硬性协议规则和可分离的生成器分布，不是生产环境的准确率——快速模式下整体约 20% 的攻击检出率应视为下限，而非准确率声明。

#### Fail-closed 行为

当所有 ML 检测器都损坏或未训练时，流水线会抛出 `DetectionUnavailable` 并丢弃所有规则引擎未做决定的数据包。这是有意为之：在异常检测器不可用时，静默的网络中断比放行未知流量更安全。状态 API 暴露了 `detection_unavailable_drops` 和 `broken_detectors`，运维人员可以据此发现该状态。

---

## 部署

### Docker

```bash
# 构建并启动
bash deploy.sh build
bash deploy.sh start

# 验证测试通过（在容器内运行全部 verify_* 脚本）
bash deploy.sh test

# 查看日志
bash deploy.sh logs

# 停止
bash deploy.sh stop
```

**说明：**
- `deploy.sh` 运行 8 个验证脚本（`verify_engine_module`、`verify_interception_module`、`verify_block_lifecycle`、`verify_live_exposed_bugs`、`verify_fpr_regression`、`verify_features_module`、`verify_data_module`、`verify_management_plane`），而不是 `pytest`。
- 容器出于安全考虑以非 root 用户 `nips` 运行。请从宿主机 bind-mount `rules.json`——它在 `docker compose up` 之前就必须存在，否则会报 `IsADirectoryError`。
- **容器只承载管理面。** NFQUEUE 与 iptables 需要宿主网络栈，因此拦截器跑在宿主机上（`cli.py start` 或 systemd 单元）。只部署容器，你得到的是一个能查看和修改规则、但不丢弃任何流量的 API——那是仪表盘，不是防御系统。
- `rules.json` 只包含**持久化**的黑名单条目（运维添加的 + 升级产生的永久封禁）。临时封禁镜像存在于临时层，从不落盘。

### Linux 宿主机

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 API 服务
python app.py

# 或直接使用 CLI（实时拦截需要 root）
sudo python cli.py start
```

### systemd

`deploy/systemd/` 下有两个单元，改好 `/opt/nips` 路径后作为 `nips-api.service` 与 `nips-interceptor.service` 安装（建议用 `systemctl edit` 的 drop-in 覆盖，不要直接改发行文件）：

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nips-api nips-interceptor
```

两者刻意分开：只有拦截器持有会改防火墙的权限，面向网络的那个进程不持有。拦截器需要真正的 root（`cli.py start` 会检查 `geteuid()`），只给 capability 不满足它。停止行为很关键——SIGTERM 处理函数会撤掉 NFQUEUE 重定向，所以 `TimeoutStopSec` 给得很宽；提前杀掉它会让内核重定向留在原位而无人消费队列，直到 nfqueue 超时才恢复，等于自己制造一次断网。

### 远程管理

API 监听在 `api.host`/`api.port`，说的是明文 HTTP。它只有一个共享 token、没有按用户身份，因此设计上应当放在做 TLS 终结、并额外校验源地址或客户端证书的反向代理之后，而不是直接暴露。随附的 docker-compose 绑定 `127.0.0.1:8000` 也是同样理由。

CLI 默认指向 `http://127.0.0.1:8000`，可每次指定或用环境变量改：

```bash
python cli.py --url https://nips.internal:8443 --token "$NIPS_API_TOKEN" status
NIPS_API_URL=http://10.0.0.5:8000 python cli.py alerts --last 20
```

---

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 分层设计、数据流、模块边界、红线
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR 工作流、提交前检查清单、我们不接受的内容
- [CODE_STYLE.md](CODE_STYLE.md) — 编码规范、导入规则、系统调用校验
- [SECURITY.md](SECURITY.md) — 漏洞报告、部署最佳实践
- [CHANGELOG.md](CHANGELOG.md) — 发布历史
- API 端点：见上文"运行 API"一节（/docs、/redoc 和 OpenAPI 在生产环境中关闭）

---

## 联系方式

- **作者**：梓铭
- **邮箱**：2147514473@qq.com
- **Issues**：[GitHub Issues](https://github.com/zimingttkx/Network-Security-Based-On-ML/issues)

## 许可证

MIT — 详见 [LICENSE](LICENSE)
