# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2026-07-30

### Rewrite

Complete project restructure from phishing URL classification demo to a real Network Intrusion Prevention System.

- Four-layer architecture: interception → features → engine → management
- NFQUEUE + iptables real-time traffic interception (Linux)
- AfterImage 115-dim incremental statistics + KitNET anomaly detection (Kitsune, NDSS'18)
- LUCID CNN-based DDoS flow detection (IEEE TNSM 2020)
- Rule engine: IP blacklist/whitelist, protocol filter, rate limiting
- DetectionPipeline with short-circuit semantics
- REST API + CLI + lightweight status page
- CI: keyword scan for simulation code, import check, smoke test

### Removed

All old simulation/demo code:
- Protection service (memory-only state machine with no OS blocking)
- Firewall module (in-memory classifier, no iptables integration)
- URL feature extractor (phishing detection, 30 fixed features)
- Traffic simulator (synthetic HTTP log generator)
- Demo algorithms script (random data generators)
- Benchmarks directory (attack simulation scripts)
- RL/DL/ML engine dead code (not wired to NIPS pipeline)
- Old stats module, training pipeline, data ingestion components
- Old templates (predict, protection, dashboard, training, model select)
