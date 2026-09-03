# NIPS — Network Intrusion Prevention System

**English** · [简体中文](README.zh-CN.md)

A server-side IPS that intercepts traffic on Linux, scores each packet through a rule engine plus an anomaly detector, and drops malicious packets via iptables.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.17+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> Before contributing, read [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md). CI rejects simulation/mock code in `networksecurity/`.

---

## How it works

```
Incoming Traffic
      |
      v
[Rule Engine] ------> BLOCK  (blacklist, rate limit, protocol filter)
      | pass
      v
[Kitsune] ----------> BLOCK  (AfterImage + KitNET anomaly detection)
      | pass
      v
[ALLOW]
```

The rule engine handles known-bad traffic deterministically (blacklist, whitelist, rate limit, protocol allowlist). Anything that passes is scored by Kitsune, an unsupervised packet-level anomaly detector that trains on normal traffic and flags deviations by reconstruction error (RMSE).

LUCID (a CNN-based DDoS detector) is **optional**. It is not loaded into the pipeline by default — it requires a trained TensorFlow model and must be explicitly enabled. See `networksecurity/engine/lucid/`.

### Algorithms

- **Kitsune (NDSS'18)** — AfterImage incremental statistics (100 features) + a KitNET autoencoder ensemble. Trains online, no labels needed.
- **LUCID (IEEE TNSM 2020)** — 1D CNN over 10-packet flow windows (11 features/packet). Off by default; needs a trained model.

> **Note on protocol filtering:** the rule engine's protocol allowlist is TCP(6) and UDP(17) only. Any other protocol — including **ICMP(1)** — is blocked by default. This means legitimate ICMP (ping, PMTUD, traceroute) is also dropped unless its source is whitelisted. If you run on a network that relies on ICMP, either whitelist the relevant sources or constrain the policy before enabling live interception.

---

## Quick Start

### Requirements

- Python 3.12+
- Linux for live interception (nfqueue + iptables, root required)
- macOS / other platforms for development and offline pcap testing

### 1. Clone

```bash
git clone https://github.com/zimingttkx/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML
```

### 2. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the API

```bash
python app.py
# API docs at http://localhost:8000/docs
```

### 4. CLI

```bash
python cli.py start                  # start live interception (Linux, root)
python cli.py stop                   # stop live interception (via API)
python cli.py status                 # engine status
python cli.py block 1.2.3.4          # block an IP
python cli.py unblock 1.2.3.4        # unblock an IP
python cli.py whitelist 10.0.0.0/8   # whitelist a subnet
python cli.py rules                  # list blacklist/whitelist entries
python cli.py alerts --last 20       # show recent alerts (via API)
python cli.py test --pcap sample.pcap  # offline detection test (no root needed)
```

#### Configuration

`config/config.yaml` drives both the engine and live interception:

```yaml
interception:
  nfqueue_num: 0
  safe_ips:                 # IPs that are never blocked (loopback is protected)
    - "127.0.0.1"
    - "::1"
engine:
  kitsune:
    fm_grace_period: 5000   # feature-mapping training packets
    ad_grace_period: 50000  # anomaly-detector training packets
    threshold_percentile: 99.0
  rule_engine:
    allowed_protocols: [6, 17]   # TCP, UDP; everything else blocked
    rate_limit:
      window_seconds: 1.0
      max_connections_per_window: 100
```

On `engine/start` the API/CLI load `safe_ips` and `nfqueue_num` from this file and pass them to the interceptor, so operator-tuned values are actually applied at runtime.

---

## API Reference

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/status` | Engine status, detectors, blocked IPs |
| `GET` | `/api/v1/stats/overview` | Traffic and blocking statistics |
| `GET` | `/api/v1/alerts` | Recent alert log (paginated) |
| `GET` | `/api/v1/rules` | Current blacklist and whitelist |
| `POST` | `/api/v1/rules/blacklist` | Add IP to blacklist |
| `DELETE` | `/api/v1/rules/blacklist/{ip}` | Remove IP from blacklist |
| `POST` | `/api/v1/rules/whitelist` | Add IP/CIDR to whitelist |
| `DELETE` | `/api/v1/rules/whitelist/{ip}` | Remove IP from whitelist |
| `POST` | `/api/v1/engine/start` | Start live interception (Linux, root) |
| `POST` | `/api/v1/engine/stop` | Stop interception and clean up iptables |

Full interactive documentation at `/docs`.

---

## Layout

```
app.py                         # FastAPI application entry point
cli.py                         # CLI management tool
config/
  config.yaml                  # Engine/interception configuration
templates/                     # Web status page templates
networksecurity/
  engine/                      # Detection engine
    detector.py                # BaseDetector interface + PacketInfo
    verdict.py                 # Verdict, Action, ThreatLevel types
    pipeline.py                # DetectionPipeline (multi-stage chain)
    rule_engine.py             # IP blacklist/whitelist, rate limiting
    kitsune/                   # Kitsune anomaly detector (NDSS'18)
      afterimage.py            # 100-dim incremental statistics
      kitnet.py                # Autoencoder ensemble
      kitsune.py               # Orchestrator
      detector_adapter.py      # BaseDetector adapter
    lucid/                     # LUCID DDoS detector (IEEE TNSM 2020, optional)
      cnn.py                   # 1D CNN model
      dataset_parser.py        # Flow buffer and feature extraction
      detector.py              # Orchestrator
      detector_adapter.py      # BaseDetector adapter
  interception/                # Linux traffic interception
    nfqueue_handler.py         # NFQUEUE binding and packet capture
    packet_parser.py           # Raw IPv4 packet parser
    iptables.py                # iptables rule management
    interceptor.py             # Live interceptor (nfqueue + pipeline)
  features/                    # Feature extraction
    flow_extractor.py          # Per-flow statistical features
    feature_registry.py        # Feature set registry
  data/                        # Data loading
    dataset_loader.py          # NSL-KDD, CICIDS2017, UNSW-NB15
    pcap_loader.py             # PCAP file reader
scripts/                       # Benchmarks & evaluation
  benchmark.py                 # Throughput + rule-engine accuracy
  benchmark_nslkdd.py          # NSL-KDD detection benchmark
  attack_simulation.py         # Large-scale attack simulation
```

---

## Live Interception (Linux only)

```bash
# 1. Install nfqueue library
pip install NetfilterQueue

# 2. Run with root privileges
sudo python -c "
from networksecurity.interception import Interceptor
from networksecurity.engine import DetectionPipeline
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector

pipeline = DetectionPipeline()
pipeline.add_detector(KitsuneDetector())

interceptor = Interceptor(pipeline)
interceptor.start()  # Blocks. Ctrl+C to stop.
"
```

The interceptor:
- Installs iptables rules to redirect traffic into NFQUEUE
- Leaves loopback traffic untouched — everything arriving on `lo` is ACCEPTed before the NFQUEUE rules, and loopback sources (`127.0.0.0/8`, `::1`) are never eligible for a permanent block (host-local traffic cannot be an attacker; blocking the DNS stub `127.0.0.53` would silently break host DNS)
- Leaves SSH (port 22) untouched
- Mirrors every ML/rule-engine BLOCK into the rule-engine blacklist, so blocks survive restarts (`rules.json`) and are re-applied to the kernel on the next start
- Removes all of its iptables rules on shutdown

`Interceptor` reads `safe_ips` and `nfqueue_num` from `config.yaml`, so the `safe_ips` list silently has no effect if the config is missing — keep `config.yaml` present and committed.

A detection timeout drops only the in-flight packet (fail-closed); it never commits a permanent block, so a slow verdict cannot ban a legitimate IP.

---

## Training dataset preparation

`DatasetLoader` (`networksecurity/data/dataset_loader.py`) loads NSL-KDD, CICIDS2017, and UNSW-NB15 as **labeled CSV** for supervised training of LUCID/Kitsune. It assumes each file is already a **header-bearing CSV** with the dataset's standard column names — it does **not** detect or convert headers, nor does it handle the raw headerless NSL-KDD `.txt` distribution. Preparing the files is the user's responsibility before calling `DatasetLoader`.

Required layout per dataset:

| Dataset | Expects | Notes |
| --- | --- | --- |
| **NSL-KDD** | CSV with header, 43 columns: 41 features in the standard NSL-KDD order, then `difficulty`, then `label` | The official `KDDTrain+.txt` / `KDDTest+.txt` are **headerless** — add the 41 standard feature names + `difficulty` + `label` before loading. Binary label: `normal`/`normal.` → 0 (benign), anything else → 1 (attack). |
| **UNSW-NB15** | CSV with header, binary `label` column (0/1), plus `id` and `attack_cat` metadata | `attack_cat` is dropped automatically (it leaks the label). |
| **CICIDS2017** | CSV with header, `Label` column (capital L), plus `Flow ID` / `Timestamp` / `Source IP` / `Destination IP` | Those four metadata columns are dropped automatically. `BENIGN` → 0, everything else → 1. |

Categorical columns are one-hot encoded (`get_dummies`, `drop_first`), missing values filled with 0, and the result is returned as `float32`. For aligned train/test encodings use `train_test_split()`, which fits the encoding on the training split and reindexes the test split to the same columns.

---

## Benchmarks

Two scripts measure behavior on your own hardware — numbers below are not validated across environments and will vary:

- `scripts/benchmark.py` — trains Kitsune on synthesized normal traffic, then reports rule-engine accuracy, training/detection throughput, and attack detection rate.
- `scripts/benchmark_nslkdd.py` — downloads NSL-KDD, maps flow records to synthetic packets, trains Kitsune on normal flows, and reports precision/recall/FPR.

Why detection on NSL-KDD is weak here: NSL-KDD records are **flow-level summaries**, not packet captures. Mapping each flow to a few packets throws away the timing and burst patterns that Kitsune learns from. Volumetric attacks (DoS, probe) survive the mapping better than content attacks (R2L, U2R), which look like ordinary TCP at the packet level. Treat the per-attack numbers as a statement of that limitation, not a measured accuracy claim.

The rule engine itself is exact: blacklist/whitelist, protocol filtering, and rate limiting are deterministic and always applied before the ML stage.

### Offline testing with real traffic

Two paths exercise the detection pipeline **without** root or iptables — useful for verifying behavior on real captures:

- **Real pcap (recommended for true performance):** capture packets and run them through the pipeline offline.
  ```bash
  # capture 30s of live traffic (requires root for the sniff)
  sudo python -c "from scapy.all import sniff, wrpcap; wrpcap('cap.pcap', sniff(iface='en0', timeout=30))"
  # offline detection — no root needed
  python cli.py test --pcap cap.pcap
  ```
  This surfaces the **real** false-positive rate (e.g. legitimate ICMP being blocked by the protocol filter), which the synthetic sim below does not. Note Kitsune needs ~55k normal packets before it leaves training mode, so short captures mostly exercise the rule engine.
- **Synthetic attack simulation:** `scripts/attack_simulation.py` generates labeled traffic and reports per-attack detection rates. Its ICMP/SSH results reflect the hard protocol rule and a separable generator distribution, not production accuracy — treat the overall ~20% attack detection in fast mode as a floor, not a claim.

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — layer design, data flow, module boundaries, red lines
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR workflow, pre-submission checklist, what we reject
- [CODE_STYLE.md](CODE_STYLE.md) — coding conventions, import rules, system call validation
- [SECURITY.md](SECURITY.md) — vulnerability reporting, deployment best practices
- [CHANGELOG.md](CHANGELOG.md) — release history
- API reference: `http://localhost:8000/docs` (Swagger)

---

## Contact

- **Author**: 梓铭
- **Email**: 2147514473@qq.com
- **Issues**: [GitHub Issues](https://github.com/zimingttkx/Network-Security-Based-On-ML/issues)

## License

MIT — see [LICENSE](LICENSE)
