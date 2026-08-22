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

- **Kitsune (NDSS'18)** — AfterImage incremental statistics (115 features) + a KitNET autoencoder ensemble. Trains online, no labels needed.
- **LUCID (IEEE TNSM 2020)** — 1D CNN over 10-packet flow windows (11 features/packet). Off by default; needs a trained model.

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
python cli.py test --pcap sample.pcap  # offline detection test
```

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
      afterimage.py            # 115-dim incremental statistics
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
- Leaves SSH (port 22) and loopback untouched
- Removes all of its iptables rules on shutdown

---

## Benchmarks

Two scripts measure behavior on your own hardware — numbers below are not validated across environments and will vary:

- `scripts/benchmark.py` — trains Kitsune on synthesized normal traffic, then reports rule-engine accuracy, training/detection throughput, and attack detection rate.
- `scripts/benchmark_nslkdd.py` — downloads NSL-KDD, maps flow records to synthetic packets, trains Kitsune on normal flows, and reports precision/recall/FPR.

Why detection on NSL-KDD is weak here: NSL-KDD records are **flow-level summaries**, not packet captures. Mapping each flow to a few packets throws away the timing and burst patterns that Kitsune learns from. Volumetric attacks (DoS, probe) survive the mapping better than content attacks (R2L, U2R), which look like ordinary TCP at the packet level. Treat the per-attack numbers as a statement of that limitation, not a measured accuracy claim.

The rule engine itself is exact: blacklist/whitelist, protocol filtering, and rate limiting are deterministic and always applied before the ML stage.

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
