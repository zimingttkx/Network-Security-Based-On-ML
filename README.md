# NIPS — Network Intrusion Prevention System

Real-time network intrusion detection and prevention using machine learning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.17+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> **Contributors**: read [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before submitting code. All PRs are checked for simulation code via CI.

---

## Overview

NIPS is a server-side network intrusion prevention system. It intercepts incoming traffic, extracts statistical features from network flows, and classifies each packet as benign or malicious using a multi-stage detection pipeline. Malicious traffic is blocked at the kernel level via iptables.

### Detection Pipeline

```
Incoming Traffic
      |
      v
[Rule Engine] ------> BLOCK (blacklist, rate limit, protocol filter)
      | pass
      v
[Kitsune] ----------> BLOCK (AfterImage + KitNET anomaly detection)
      | pass
      v
[LUCID] ------------> BLOCK (CNN-based DDoS flow detection)
      | pass
      v
[ALLOW]
```

### Integrated Algorithms

- **Kitsune (NDSS'18)** — AfterImage incremental statistics (115 features) + KitNET autoencoder ensemble for online anomaly detection. Unsupervised, low-latency.
- **LUCID (IEEE TNSM 2020)** — Lightweight 1D CNN for real-time DDoS detection. 10 packets per flow window, 11 features per packet.
- **ML Classifiers** — RandomForest, XGBoost, and ensemble methods for supervised flow classification using NSL-KDD / CICIDS2017.

---

## Quick Start

### Requirements

- Python 3.12+
- Linux (for live interception with nfqueue/iptables)
- macOS (for development and offline testing)

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
python cli.py status          # engine status
python cli.py block 1.2.3.4   # block an IP
python cli.py unblock 1.2.3.4 # unblock an IP
python cli.py whitelist 10.0.0.0/8  # whitelist a subnet
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

Full interactive documentation at `/docs`.

---

## Architecture

```
app.py                         # FastAPI application entry point
cli.py                         # CLI management tool
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
    lucid/                     # LUCID DDoS detector (IEEE TNSM 2020)
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

The interceptor automatically:
- Sets up iptables rules to redirect traffic to NFQUEUE
- Protects SSH (port 22) and loopback
- Cleans up all iptables rules on shutdown

---

---

## Benchmark

Benchmarked on NSL-KDD, the standard NIDS dataset from UNSW and the Canadian Institute for Cybersecurity (125,973 training flows, 11,849 test flows). Flows were mapped to per-packet `PacketInfo` objects and processed through the Kitsune pipeline (AfterImage 115-dim features + KitNET autoencoder ensemble).

Test environment: GitHub Codespaces (2 vCPU, 8 GB RAM).

### Kitsune — Unsupervised Anomaly Detection

| Metric | Value |
| ------ | ----- |
| Training packets | 150,000 |
| Training throughput | 819 pkt/s |
| Detection throughput | 1,126 pkt/s |
| Sustained throughput | 1,085 pkt/s |
| Precision | 89.0% |
| False Positive Rate | 3.2% |

### Per-Attack Detection Rate

| Attack Category | Detection Rate | Notes |
| --------------- | -------------- | ----- |
| DoS (SYN flood, Neptune, Smurf) | 15% | Volumetric — packet-level burst patterns partially detectable |
| Probe (port scan, IP sweep) | 9% | Low-rate — mapping flows to packets loses scan cadence |
| R2L (password guess, warezclient) | <1% | Content-level — indistinguishable from normal TCP at packet level |
| U2R (buffer overflow, rootkit) | <1% | Content-level — AfterImage sees normal-sized packets with normal flags |

### Interpretation

Kitsune is an **unsupervised packet-level** detector. 89% precision means when it flags something, it is almost certainly malicious. The 3.2% FPR means normal traffic is rarely misclassified — acceptable for a NIPS in blocking mode.

The low recall (especially on R2L and U2R) reflects a fundamental limitation of this benchmark: NSL-KDD records are **flow-level summaries**, not real packet captures. R2L/U2R attacks look identical to normal traffic at the per-packet level. DoS and probe attacks show more promise because their volumetric patterns survive the flow→packet mapping. Real per-packet detection accuracy on live pcap traces is expected to be significantly higher for DoS and probe categories.

### Rule Engine — Deterministic Filtering

The rule engine provides microsecond-level, 100% accurate filtering for known IPs (blacklist/whitelist), protocol filtering, and rate limiting. Combined with Kitsune anomaly detection, this provides defence-in-depth: fast rule-based pre-filtering followed by ML-based anomaly detection for unknown threats.

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

---

## Star History

<a href="https://www.star-history.com/?repos=zimingttkx%2FNetwork-Security-Based-On-ML&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=zimingttkx/Network-Security-Based-On-ML&type=date&legend=top-left" />
 </picture>
</a>
