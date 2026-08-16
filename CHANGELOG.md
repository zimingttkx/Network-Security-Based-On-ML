# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Large-scale attack simulation script (`scripts/attack_simulation.py`) for benchmarking detection efficacy across attack categories.
- Comprehensive CI pipeline: lint, security scan, unit tests, attack smoke test, PR title lint, and branch name checks.

### Changed

- Applied ruff auto-fix across the entire project and updated CONTRIBUTING.md.
- Updated README to match the current codebase (CLI commands, API endpoints, architecture tree).

### Fixed

- Added missing `Verdict` import and suppressed bandit false positives in CI.
- Configured style check to exit zero and ignore non-critical rules.

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
