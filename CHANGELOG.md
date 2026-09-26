# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Management-API token authentication: when `api.auth_token` (or the `NIPS_API_TOKEN` env var) is set, every `/api/v1/*` route requires the `X-API-Token` header; an empty token disables auth with a startup warning. Wildcard CORS was removed in favor of an explicit origin allowlist.
- BLOCK-verdict escalation policy (`engine/block_policy.py`): strikes per source IP inside a rolling window escalate to a temp ban (kernel DROP + in-memory blacklist entry with a TTL, auto-lifted by an expiry sweeper) and, after repeated temp bans, to a permanent ban persisted to `rules.json`. A single anomaly no longer installs a permanent kernel DROP.
- `GET /api/v1/blocks` endpoint exposing the live escalation state, plus `kernel_blocked_ips` and `detection_loop_stale_seconds` fields on `/api/v1/status`.
- `broken_detectors` (circuit-breaker state) exposed on `/api/v1/status`, as documented in SECURITY.md.
- Parquet support in `DatasetLoader`: `.parquet` files are read as Parquet; CSV remains the default.
- Detector contract: `BaseDetector` gains `configure()`, `ready` and `status()`, and mounting is driven by config (`engine.ml.enabled`, `engine.ml.detectors`) instead of hardcoded wiring in the entrypoints. `networksecurity/engine/threshold_detector.py` is a complete worked example; with the switch off, no ML module is imported at all, so deleting `engine/kitsune/` and `engine/lucid/` yields a supported rules-only deployment.
- `scripts/verify_detector_contract.py`: contract, switch, external mount, skipped-on-failure, and config-degradation checks.
- Per-kind event-store write counters (`written_alerts` / `written_audit`) in `stats()`; the store-wide `written` total keeps its meaning, so `nips_alert_events_written_total` is unchanged.
- `requirements-dev.txt` separating development/CI dependencies (Parquet engine, scapy) from the runtime install.
- `preflight` gate that imports every production module in an environment built from `requirements.txt` alone — the class of gap where a dependency exists only in the CI install list.
- Real-data evaluation scripts: UNSW-NB15 pcap reconstruction (`scripts/build_unsw_pcap.py`) and end-to-end per-category evaluation (`scripts/evaluate_pcap.py`); cross-module regression scripts (`scripts/verify_*.py`) including a post-training FPR regression assertion in CI (`scripts/verify_fpr_regression.py`).
- Large-scale attack simulation script (`scripts/attack_simulation.py`) for benchmarking detection efficacy across attack categories.
- Comprehensive CI pipeline: lint, security scan, unit tests, FPR regression guard, PR title lint, and branch name checks.

### Changed

- Learning detection is off by default (`engine.ml.enabled: false`). The distinction is now explicit: switching it off is a decision, so packets the rule engine does not decide are allowed; having it on and unable to run is an outage, so those packets are dropped as before.
- A detector mounted without its model (`ready: false`) is no longer registered and no longer counts as ML coverage. It used to answer `LOG`, which both ended the chain and made the pipeline believe detection had run — enough for a tripped live detector behind it to disable fail-closed silently.
- `/api/v1/status` passes `pipeline.status()` through instead of copying fields one by one, and splits detector state into `ml_enabled` / `ml_consulted` / `ml_idle` plus a per-detector `detector_status` (a detector's own `status()` output, contained if it raises), so "not deployed", "could not run", "warming up" and "running" are no longer the same sentence.
- README's `## Benchmarks` became `## Measured results and limits`: every number now carries the command that reproduces it.
- Real-capture evaluation now trains at the shipped grace periods (5 000 + 50 000, `config/config.yaml`) instead of a shortened 2 000 + 30 000 that fit training and a detection window into one capture. The override also fitted KitNET's feature map on 2 000 packets — whatever arrived first — which made the measured operating point a property of the packet draw: redrawing the same 1 680 flows moved it from 1.3% to 52% of packets blocked with no code change, and at 2 000 packets the fitting window covers 43 flows with one of them supplying a third of it. At the shipped values both draws stay out of that regime; the READMEs quote the measured range.
- The real-capture quality gate scores three fixed draws (the low, middle and high of a twelve-seed sweep) and bounds their median false-positive rate instead of bounding a single unseeded run, plus a wider bound that no draw is catastrophic. KitNET draws its autoencoder weights from the global RNG, which on this capture spreads the false-positive rate from 2.1% to 10.5% across seeds — the old single-run 5% bound sat inside that spread and would fail roughly one run in five with nothing changed. Fixed seeds make the gate reproducible (`--seed` in `evaluate_pcap.py` pins a draw bit-for-bit, verified by rerunning one). It also fails when no draw flags a single attack packet, the dead-detector case its docstring described but never checked.
- CI restructured into three blocking layers (static / unit / system) behind one aggregate `ci/required` check, so branch protection pins a single name and skipped jobs no longer block documentation-only PRs; the eight module suites moved from serial steps in one job to a parallel matrix. Slow calibration-only checks (real-capture detection quality, LUCID training, attack simulation), the Python 3.13 sweep and the full static reports moved to `.github/workflows/nightly.yml`.
- `EventStore.flush()` waits for the in-flight batch to commit, not just for the queue to drain, so `stats()` read straight after a flush cannot report a total that has not caught up.
- `DatasetLoader` reports a missing Parquet engine as an actionable `ImportError` instead of surfacing pandas' internal one.
- TensorFlow moved to the `nips[lucid]` extra: default installs no longer pull in 500 MB+ of dependencies; the LUCID adapter stays inactive without it (lazy import).
- Config loading reworked: `config.yaml` now drives the `api` block (host/port, auth token, CORS), the `engine` block, and the `blocking` (escalation) policy; paths are package-root anchored so the CWD no longer matters, and fallback defaults include `::1` in `safe_ips`.
- Detection enforcement is fail-closed: detection errors and timeouts drop only the in-flight packet and never commit a permanent block.
- Applied ruff auto-fix across the entire project and updated CONTRIBUTING.md.
- Updated README to match the current codebase (CLI commands, API endpoints, architecture tree).

### Removed

- Web status page: the `/` dashboard route, `templates/` (`index.html`), and the static-file mount are gone — the management surface is now the REST API + CLI only. `jinja2` and `python-multipart` were dropped from the dependencies.

### Fixed

- Kitsune: removed AfterImage wall-clock features that caused 100% false positives on long-running sessions; fixed KitNET output-layer normalization ordering; wired `threshold_percentile`; `is_ready` exposed as a property.
- Rule engine: O(n) full-table scan replaced with O(1) LRU eviction in the rate limiter; counters and status snapshots taken under lock; rate-limit cap raised and eviction clarified.
- Interception: loopback self-banning prevented, ban state kept in sync across layers, NFQUEUE queue numbers aligned, shutdown/close races and cross-thread data races fixed, fail-open hole closed; `Interceptor.unblock_ip` and the unblock endpoint now also clear kernel-level bans.
- Data: PcapLoader IP-parsing scope fix (real pcaps no longer parse every IP to None); pcap loader rewritten to mirror the live parser and support more link layers; scapy import deferred to call time; Infinity values no longer leak through Parquet datasets.
- API/app: rule-entry IP/CIDR validation, atomic `rules.json` writes, absolute rules path, alerts list lock, engine config wired into the API/CLI start path, undefined `RULES_FILE_DEFAULT` reference.
- Added missing `Verdict` import and suppressed bandit false positives in CI.
- `scripts/attack_simulation.py`: the mixed-traffic phase kept its own tallies, so the summary table — and the headline false-positive rate — silently excluded all of its packets. On an interleaved run that dropped 60,000 verdicts including a measured ~100% false-positive rate on normal traffic, which the report then printed as a fraction of a percent. Phase 3 now feeds the same counters as every other phase, and the report prints a reconciliation warning whenever the summary stops matching the pipeline's own block count.
- The bundled "real capture" benchmark no longer writes its answer key into the capture. `build_unsw_pcap.py` drew attack flows' source addresses from one block and normal traffic's from another, and `evaluate_pcap.py` recovered the label by reading that block back out of the header, so a detector keyed on the source address scored 100% for free and the reported detection rate measured the reconstruction. Addresses are now drawn independently of the label, labels live in a sidecar (`scripts/capture_labels.py`) that the evaluator takes as an explicit input, and a capture without one is refused rather than scored (`--no-labels` reports verdicts and block reasons only). `scripts/verify_real_capture_quality.py` gates the three ways the leak could come back: a permutation test that the source address cannot predict the label (self-tested on a deliberately leaky partition — a check that cannot fail is not a check), the retired `175.45.176.*`-means-attack rule, and two evaluator runs — one with a missing label file, one with `--no-labels` — that must refuse to invent ground truth.
- Style and security findings that cannot fail the build are now reported as such instead of running behind an `--exit-zero` that made them look like passing gates.

### Performance

- RateLimiter over-limit throughput 736k → 953k checks/s (O(1) LRU eviction); FlowTracker high-entropy flood throughput 412 → 257,799 pkt/s (LRU + heap expiry sweep) with bounded memory.

## [1.0.0] — 2026-07-30

### Rewrite

Complete project restructure from phishing URL classification demo to a real Network Intrusion Prevention System.

- Four-layer architecture: interception → features → engine → management
- NFQUEUE + iptables real-time traffic interception (Linux)
- AfterImage 100-dim incremental statistics + KitNET anomaly detection (Kitsune, NDSS'18)
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
