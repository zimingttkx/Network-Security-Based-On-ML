# NIPS — Network Intrusion Prevention System

**English** · [简体中文](README.zh-CN.md)

A server-side IPS that intercepts traffic on Linux, scores each packet through a rule engine plus an anomaly detector, and drops malicious packets via iptables.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> Before contributing, read [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md). CI rejects simulation/mock code in `networksecurity/`.

---

## How it works

```
Incoming Traffic
      |
      v
[Rule Engine] ------> decides: blacklist, whitelist, protocol filter,
      | abstain       rate limit, signatures   (always present)
      v
[Mounted learning detectors] ------> BLOCK on anomaly
      |
      v
[ALLOW]
```

The rule engine is the permanent part of the chain: deterministic, and the only stage that runs on a default configuration. Traffic it does not decide falls through to whatever detectors you mount.

The learning detectors are a **plug-in**, and they are **off by default**:

```yaml
engine:
  ml:
    enabled: false        # true turns detection on for whatever is listed below
    detectors:
      - uses: kitsune     # or `lucid`, or `my_package.module:MyDetector`
```

With `enabled: false`, no ML module is imported or constructed: deleting `networksecurity/engine/kitsune/` and `networksecurity/engine/lucid/` leaves a working, rules-only deployment. With it on, a detector that cannot be built is logged and skipped rather than taking the process down. The interface a third-party detector implements (`process_packet` / `configure` / `ready` / `status`) and the fail-closed rules that follow from the switch are in [ARCHITECTURE.md — The Detector Contract](ARCHITECTURE.md); `networksecurity/engine/threshold_detector.py` is a complete example to copy.

What detection here does and does not achieve is measured, not asserted: see [Measured results and limits](#measured-results-and-limits).

LUCID (a CNN-based DDoS detector) is optional in its own right as well: it needs TensorFlow (`pip install -e ".[lucid]"` or `pip install tensorflow`) **and** a trained model at `engine.lucid.model_path`. Without a model that loads, it is not mounted at all — it no longer sits in the chain as an inactive detector the status page would still list.

### Algorithms

- **Kitsune (NDSS'18)** — AfterImage incremental statistics (90 features) + a KitNET autoencoder ensemble. Trains online, no labels needed. When link-layer headers are absent (live NFQUEUE), the MAC channel uses a `(protocol, ttl)` proxy key so it never collapses to zero variance. Grace periods (`fm_grace_period`, `ad_grace_period`) allow warmup before detection starts; during this time packets are logged but not blocked.
- **LUCID (IEEE TNSM 2020)** — 1D CNN over 10-packet flow windows (11 features/packet). Off by default; needs a trained model and `engine.lucid.model_path` set in config. Produce the model with `scripts/train_lucid.py` (see "Training LUCID" below).

> **Note on protocol filtering:** the rule engine's protocol allowlist is TCP(6) and UDP(17); anything else it inspects is blocked, including **ICMP(1)**. In live interception, however, only TCP and UDP are redirected into NFQUEUE (`interception.intercept_icmp` is off by default) — so there ICMP is **not inspected and not blocked**: the host's own firewall decides. Turn `interception.intercept_icmp: true` on to bring ICMP into the pipeline, then allow individual types through `engine.rule_engine.allowed_icmp_types` — blocking the whole protocol also breaks Path MTU Discovery (type 3, "frag needed"), which blackholes large connections, so a type list is the useful setting rather than an all-or-nothing ban. **ICMPv6(58)** has one exception to that list: the types that maintain the link itself — neighbour and router solicitation/advertisement (135/136/133/134) and "packet too big" (2) — always pass, because a host whose neighbour discovery is queued and dropped has not blocked an attacker, it has taken itself off the network. ICMPv6 echo is *not* in that set and stays behind the protocol filter (allow-by-type covers ICMPv4 above). Offline pcap runs (`cli.py test --pcap`) do exercise the protocol filter, since those packets reach the engine whatever their protocol.

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

# Development / CI — adds pyarrow (the bundled datasets are Parquet) and scapy
# (offline pcap testing).  The runtime image installs neither.
pip install -r requirements-dev.txt

# Optional: the LUCID CNN detector needs TensorFlow, which is not part of the
# default install (the adapter stays inactive without it).
pip install -e ".[lucid]"     # or: pip install tensorflow
```

### 3. Configure

`config/config.yaml` drives both the engine and live interception:

- `engine.ml.enabled` / `engine.ml.detectors`: whether learning detection runs at all, and which detectors to mount (built-in short names, or `package.module:ClassName` with a `params:` block). Off by default.
- `engine.kitsune.*`: grace periods, threshold percentile, learning_rate (passed to AfterImage)
- `engine.lucid.model_path`: set a path to enable LUCID; empty string disables it
- `api.auth_token`: set to enable authentication; empty string disables auth (development mode)
- `api.host` / `api.port`: what `python app.py` binds to
- `interception.safe_ips`: add IPs that must never be blocked (loopback included by default)
- `interception.intercept_icmp` / `engine.rule_engine.allowed_icmp_types`: ICMP policy (see the protocol-filtering note above)
- `storage.*`: event database path, row cap and retention window (see "Alerts, audit and metrics")
- `logging.*`: level, rotating file target and syslog forwarding

### 4. Run the API

```bash
python app.py
# /docs, /redoc and the OpenAPI schema are all disabled in production.
```

### 5. CLI

```bash
python cli.py start                  # start live interception (Linux, root)
python cli.py stop                   # stop live interception (via API)
python cli.py status                 # engine status
python cli.py block 1.2.3.4          # block an IP (POST /api/v1/rules/blacklist)
python cli.py unblock 1.2.3.4        # unblock an IP (DELETE /api/v1/rules/blacklist/{ip})
python cli.py whitelist --ip 10.0.0.0/8   # whitelist a subnet (rejects /0 default routes)
python cli.py unwhitelist --ip 10.0.0.0/8 # remove from whitelist
python cli.py rules                  # list blacklist/whitelist entries
python cli.py reload                 # apply edited rules.json / config to the running engine
python cli.py alerts --last 20       # stored alerts, newest first (via API)
python cli.py alerts --source-ip 203.0.113.7 --action block
python cli.py alerts --since 2026-09-19T00:00:00 --format csv > alerts.csv
python cli.py audit --last 20        # who changed which rule, and the outcome
python cli.py audit --result 401     # rejected management attempts
python cli.py test --pcap sample.pcap  # offline detection test (no root needed)
```

#### Configuration

`config/config.yaml` drives both the engine and live interception:

```yaml
interception:
  nfqueue_num: 0
  intercept_icmp: false     # redirect ICMP into NFQUEUE so the type policy applies
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
    allowed_icmp_types: []       # ICMP types that pass despite protocol 1 not being listed,
                                 # e.g. [0, 3, 4, 8, 11] to keep PMTUD and ping alive.
                                 # ICMPv6 link maintenance always passes, whatever this says
    rate_limit:
      window_seconds: 1.0
      max_connections_per_window: 100
blocking:                    # BLOCK escalation policy (see "Live Interception")
  strikes_threshold: 5       # BLOCKs inside the window before a temp ban
  strikes_window: 300.0
  temp_ban_seconds: 600.0
  temp_ban_count_to_perm: 3  # completed temp bans before a permanent ban
api:
  auth_token: ""             # empty = auth disabled (dev only); NIPS_API_TOKEN overrides
  cors_origins:              # explicit allowlist — "*" is not supported
    - "http://localhost:8000"
```

On `engine/start` the API/CLI read the `interception`, `engine`, `blocking`, and `api` blocks from this file and apply them at runtime. If the file is missing or malformed, each loader falls back to safe defaults (loopback protection included) rather than crashing.

---

## API Reference

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/status` | Engine status, detectors, blocked IPs (incl. kernel-level), detection-loop health |
| `GET` | `/api/v1/stats/overview` | Traffic and blocking statistics |
| `GET` | `/api/v1/alerts` | Stored alerts: `limit`, `offset`, `source_ip`, `action`, `since`, `until`, `format=json\|csv\|jsonl` |
| `GET` | `/api/v1/audit` | Management audit trail: `limit`, `offset`, `actor`, `result`, `since`, `until`, `format` |
| `GET` | `/api/v1/rules` | Current blacklist and whitelist |
| `GET` | `/api/v1/blocks` | Live escalation state (observing / temp-banned / perm-banned) |
| `POST` | `/api/v1/rules/blacklist` | Add IP to blacklist |
| `DELETE` | `/api/v1/rules/blacklist/{ip}` | Remove IP from blacklist |
| `POST` | `/api/v1/rules/whitelist` | Add IP/CIDR to whitelist |
| `DELETE` | `/api/v1/rules/whitelist/{ip}` | Remove IP from whitelist |
| `POST` | `/api/v1/rules/reload` | Re-read rules.json and the live engine knobs from config.yaml |
| `GET` | `/api/v1/signatures` | Declared signature rules and their hit counts |
| `POST` | `/api/v1/signatures` | Add or edit a signature (re-posting an id replaces it) |
| `DELETE` | `/api/v1/signatures/{id}` | Remove a signature |
| `POST` | `/api/v1/engine/start` | Start live interception (Linux, root) |
| `POST` | `/api/v1/engine/stop` | Stop interception and clean up iptables |
| `GET` | `/metrics` | Prometheus text exposition (token-guarded like `/api/v1/*`) |

The two `DELETE` routes take `{ip:path}`, so CIDR entries are removable too (`/api/v1/rules/blacklist/10.0.0.0%2F8` or the unencoded form).

**Authentication:** when `api.auth_token` is set in `config.yaml` (or the `NIPS_API_TOKEN` env var is present), every `/api/v1/*` call must carry the header `X-API-Token: <token>`. An empty token disables authentication (development only — the server logs a warning at startup). `/health` stays open (liveness probes).

### Alerts, audit and metrics

Detection events and every management action are written to SQLite (WAL) at `storage.events_db` (default `data/events.db`) — they survive a restart, and `retention_days` plus `max_rows` bound the file.

- **Alerts** (`/api/v1/alerts`, `cli.py alerts`) — one row per BLOCK verdict, plus whitelist changes. `format=csv|jsonl` exports for a SIEM; each request is capped at 1000 rows.
- **Audit** (`/api/v1/audit`, `cli.py audit`) — one row per rule/engine change *and* per refused attempt (401/422), with the peer address, method, path, target and outcome. The API has one shared token, so `actor` identifies the host, not a named user.
- **Metrics** (`/metrics`) — packets processed/blocked, detector state, blacklist sizes, temp-ban counters, event-store health.

The detection path never waits on the disk: `record_alert` only enqueues into a bounded buffer and a background thread writes in batches. If the buffer overflows or a batch fails, the counters in `nips_alert_events_dropped_total` / `nips_event_store_write_errors_total` rise and `/api/v1/status` reports them under `event_store` — an incomplete trail is visible, not silent. When the database is unusable, reads fall back to the most recent 500 in-memory events and `event_store.degraded` is true.

`logging.file` adds a rotating log file and `logging.syslog_address` forwards to syslog (platform socket, or `host:port` over UDP); an unreachable target is reported and skipped rather than blocking startup, and a sink that starts failing later (daemon restarted, disk full) reports once per 1/10/100/1000 lost records instead of printing a traceback per record.

### Signature rules

A blacklist answers "is this source bad"; the rate limiter answers "is anyone sending too fast". Neither answers *"drop TCP/22 traffic from 203.0.113.0/24 once it exceeds 50 sessions a minute"* — banning the subnet would silence the legitimate users behind it, and the global limiter cannot be scoped to one source and one port. A signature is that conjunction:

```bash
# watch first: counts matches, drops nothing
python cli.py signature add --id ssh-brute --src 203.0.113.0/24 \
    --protocol tcp --dport 22 --min-packets 50 --window 60 --action log
python cli.py signature list                    # rules with their hit counts
python cli.py signature add --id ssh-brute --src 203.0.113.0/24 \
    --protocol tcp --dport 22 --min-packets 50 --window 60    # now enforce
python cli.py signature delete ssh-brute
```

| field | meaning |
|---|---|
| `src` / `dst` | IP or CIDR; a `/0` default route is refused |
| `protocol` | `tcp`, `udp`, `icmp`, or a number |
| `dport` / `sport` | 0-65535. Giving a port without a protocol defaults to TCP — those byte offsets are an echo id/sequence in ICMP, so matching them there would be nonsense |
| `tcp_flags` | exact match on the 6-bit field (`0x02` = SYN); only valid with TCP |
| `min_packets` + `window_seconds` | fire only after N matches from one source inside the window |
| `action` | `block` drops that packet inline; `log` counts it and lets it continue to the ML stage |

- A rule with **no matchers** is refused: with `action=block` a single stray call would drop all traffic.
- Evaluation is declaration order, first match wins, after the blacklist (so a listed source is reported as listed) and before the global rate limit (so a scoped rule is not masked by it).
- A signature BLOCK is enforced **inline per packet**, like any other rule-engine verdict: it does not count strikes and installs no kernel DROP. That is deliberate for rate-conditioned rules — a persistent kernel rule would outlive the condition that triggered it. To ban a source outright, use the blacklist.
- Per-source hit counters are LRU-bounded (10k sources per rule); uncapped, a spoofed flood would grow the table one entry per packet and turn a detection feature into memory exhaustion.
- Signatures persist in `rules.json` under `"signatures"` and follow the hot-reload rules above: a hand edit applies within 30 s (or via `cli.py reload`), and one malformed entry rejects the whole file rather than applying part of it.
- API: `GET`/`POST /api/v1/signatures` and `DELETE /api/v1/signatures/{id}`; validation lives in the engine, so the API, the file and hot reload refuse exactly the same specs. Upserts and removals are audited.

### Hot reload

`rules.json` and `config/config.yaml` are watched by mtime, so a running engine picks up edits within 30 s; `POST /api/v1/rules/reload` (or `cli.py reload`) applies them immediately. Restarting is not required — and would be costly, since Kitsune re-trains from zero.

- `rules.json` is applied with **replace** semantics, so deleting an entry really stops enforcing it (startup uses merge, which only adds).
- `engine.rule_engine.rate_limit.*` and `allowed_protocols` take effect on the next packet.
- A malformed file is rejected wholesale: the live rules stay exactly as they were, `/api/v1/status` raises `reload.failures`, and the attempt is audited as `reload_failed`.
- Entries the kernel would refuse anyway (loopback / `safe_ips`) are swept as at startup and reported as `dropped_unenforceable`.
- Kitsune's `fm_grace_period`, `ad_grace_period`, `threshold_percentile` and `learning_rate` are **not** re-applied — they describe how the detector was trained, so they need a restart. The reload summary names them.

---

## Layout

```
app.py                         # FastAPI application entry point
cli.py                         # CLI management tool
config/
  config.yaml                  # Engine/interception configuration
networksecurity/
  engine/                      # Detection engine
    detector.py                # BaseDetector interface + PacketInfo
    assembly.py                # Mounts the detectors config asks for (engine.ml)
    threshold_detector.py      # Complete worked example of the contract
    verdict.py                 # Verdict, Action, ThreatLevel types
    pipeline.py                # DetectionPipeline (multi-stage chain)
    rule_engine.py             # IP blacklist/whitelist, rate limiting
    block_policy.py            # BLOCK escalation: strikes → temp ban → permanent ban
    kitsune/                   # Kitsune anomaly detector (NDSS'18)
      afterimage.py            # 90-dim incremental statistics
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
    packet_parser.py           # Raw IPv4/IPv6 packet parser
    iptables.py                # iptables rule management
    interceptor.py             # Live interceptor (nfqueue + pipeline)
  features/                    # Feature extraction
    flow_extractor.py          # Per-flow statistical features
    feature_registry.py        # Feature set registry
  data/                        # Data loading
    dataset_loader.py          # NSL-KDD, CICIDS2017, UNSW-NB15 (CSV / Parquet)
    pcap_loader.py             # PCAP file reader
  observability/               # Durable events and metrics (storage only)
    alert_store.py             # SQLite alerts + audit trail, batched non-blocking writer
    metrics.py                 # Prometheus text exposition
    log_setup.py               # level / rotating file / syslog routing
  utils/                       # Shared helpers
    config.py                  # config.yaml loading (engine / api / blocking / storage / logging)
    validation.py              # IP/CIDR validation and blacklist refusal rules
scripts/                       # Benchmarks, evaluation & regression checks
  benchmark.py                 # Throughput + rule-engine accuracy
  benchmark_nslkdd.py          # NSL-KDD detection benchmark
  attack_simulation.py         # Large-scale attack simulation
  build_unsw_pcap.py           # Rebuild real-traffic pcaps from the bundled UNSW-NB15 flows
  train_lucid.py               # Train the LUCID CNN and write engine.lucid.model_path
  evaluate_pcap.py             # End-to-end pcap evaluation (per attack category)
  verify_*.py                  # Module regression checks, incl. the CI FPR guard
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
- Redirects only TCP and UDP into NFQUEUE unless `interception.intercept_icmp` is on; with it on, `allowed_icmp_types` decides which ICMP types the engine then accepts (ICMPv6 link maintenance passes regardless)
- Enforces BLOCK verdicts through an escalation policy (`blocking:` in `config.yaml`) that applies **only to ML-detector BLOCKs**. Rule-engine verdicts (blacklist hit, rate limit, protocol filter) are deterministic and already enforced inline on every packet, so they never count strikes and cannot escalate — this also guarantees an operator's blacklist entry can never be modified by the ban lifecycle. A single ML BLOCK only inline-drops that packet and counts a strike against the source. Crossing `strikes_threshold` inside the rolling window triggers a **temp ban** — kernel DROP plus a rule-engine blacklist *mirror* with a TTL, lifted automatically on expiry (only the mirror is removed; an operator's own entry is never touched). Repeated temp bans escalate to a **permanent ban**, which is mirrored into `rules.json`; on the next start it is loaded back into the rule engine and enforced per-packet in userspace — the kernel DROP itself is **not** reinstalled
- Removes all of its iptables rules on shutdown

**Dual-stack, with an honest fallback.** IPv4 *and* IPv6 TCP/UDP are redirected into NFQUEUE, parsed (including the IPv6 extension-header chain) and blocked through `ip6tables`. If `ip6tables` is unavailable the interceptor starts anyway, refuses every IPv6 block instead of pretending, and reports the gap: `ipv6_intercepted: false` in `/api/v1/status` and `nips_ipv6_intercepted 0` in `/metrics`. Check that gauge on any dual-stack host — a silently uninspected second address family is exactly the failure an operator would not notice until an incident.

Three parsing limits remain, all fail-closed and all counted in `nfqueue_parse_failed`: non-first IP fragments carry no transport header, so they are dropped rather than misread; AH/ESP packets cannot be walked without authenticating them, so they are dropped rather than parsed as if the ciphertext were a TCP header; and an IPv6 extension chain deeper than six hops is treated as crafted.
`Interceptor` reads `safe_ips` and `nfqueue_num` from `config.yaml`; a missing or unparseable file falls back to safe defaults (loopback protection included) rather than starting unprotected.

A detection timeout drops only the in-flight packet (fail-closed); it never commits a permanent block, so a slow verdict cannot ban a legitimate IP.

---

## Training dataset preparation

`DatasetLoader` (`networksecurity/data/dataset_loader.py`) loads NSL-KDD, CICIDS2017, and UNSW-NB15 as **labeled CSV or Parquet** for supervised training of LUCID/Kitsune. It assumes each file is already a **header-bearing CSV** with the dataset's standard column names (a `.parquet` suffix is read as Parquet instead) — it does **not** detect or convert headers, nor does it handle the raw headerless NSL-KDD `.txt` distribution. Preparing the files is the user's responsibility before calling `DatasetLoader`.

Required layout per dataset:

| Dataset | Expects | Notes |
| --- | --- | --- |
| **NSL-KDD** | CSV with header, 43 columns: 41 features in the standard NSL-KDD order, then `difficulty`, then `label` | The official `KDDTrain+.txt` / `KDDTest+.txt` are **headerless** — add the 41 standard feature names + `difficulty` + `label` before loading. Binary label: `normal`/`normal.` → 0 (benign), anything else → 1 (attack). |
| **UNSW-NB15** | CSV with header, binary `label` column (0/1), plus `id` and `attack_cat` metadata | `attack_cat` is dropped automatically (it leaks the label). |
| **CICIDS2017** | CSV with header, `Label` column (capital L), plus `Flow ID` / `Timestamp` / `Source IP` / `Destination IP` | Those four metadata columns are dropped automatically. `BENIGN` → 0, everything else → 1. |

Categorical columns are one-hot encoded (`get_dummies`, `drop_first`), missing values filled with 0, and the result is returned as `float32`. For aligned train/test encodings use `train_test_split()`, which fits the encoding on the training split and reindexes the test split to the same columns.

### Training LUCID

LUCID is the one detector that is supervised, so it needs a model file before `engine.lucid.model_path` can point at anything:

```bash
# 1. check what your labels imply — no TensorFlow needed, writes nothing
python scripts/train_lucid.py --pcap capture.pcap \
    --attackers 203.0.113.0/24 --victims 10.0.0.1 --inspect

# 2. fit and save (pip install -e ".[lucid]" first)
python scripts/train_lucid.py --pcap capture.pcap \
    --attackers attackers.txt --victims 10.0.0.1 --out models/lucid_cnn.h5
```

`--inspect` prints how many complete windows the capture yields, the attack/benign balance and how many flows expired before filling a window — the numbers that tell you whether the addresses you named label anything at all before you spend a training run. Notes:

- Attacker/victim entries may be single addresses or CIDRs. A malformed one is refused rather than skipped: a typo in an attacker list silently unlabels exactly the traffic the model was meant to learn.
- Under LUCID's convention a window is an attack if a **majority** of its packets involve an attacker *or* a victim address on either side — so listing a victim marks every flow toward it as attack traffic. Name attackers only if your capture contains ordinary traffic to that host.
- Training reuses the online feature path, so a model cannot be fitted on one representation and scored against another.
- One-sided labels, zero complete windows, or fewer than four windows are refused with the reason printed, instead of producing a model that predicts one class and looks healthy.

---

## Measured results and limits

These are numbers this repository actually produces, with the command that reproduces each one. They are not stable — Kitsune's projections are unseeded — so treat them as ranges.

| what | result | reproduce with |
|---|---|---|
| Rule engine vs. single-source SYN flood | **100%** detected (1000 pkt/s from a small pool trips the per-source limit), **2.4%** FPR on normal traffic | `python scripts/verify_fpr_regression.py` |
| Same flood spread over 2000 spoofed sources | **0.1%** — per-source rate never crosses the limit, and per-host modelling cannot see it | `python scripts/attack_simulation.py --full` |
| Volumetric floods on synthetic traffic (UDP, ICMP) | 100% (ICMP by protocol rule, UDP by the anomaly stage) | same |
| Whole synthetic run, every phase counted | 64.7% attack packets detected, **26.75% of normal packets blocked** | same |
| Real UNSW-NB15 reconstruction | false-positive rate **1.8–3.7%** | `python scripts/evaluate_pcap.py` |
| Real UNSW-NB15 reconstruction, detection rate | ~~0.0–0.7%~~ — **void as an accuracy claim**, see below | same |
| Offline pipeline throughput | ~660–870 pkt/s, single process | either of the above |

**The bundled "real capture" benchmark leaks its own labels.** `scripts/build_unsw_pcap.py` assigns source addresses *from the label* (`ATTACK_NET` for attack flows, `NORMAL_NET` for normal ones, since UNSW-NB15 ships no per-row IPs), and `scripts/evaluate_pcap.py` then reads the label back out of those addresses as ground truth. Any rule keyed on source address therefore scores 100% on it for free, and the detection-rate number measures the reconstruction, not the detector. The false-positive rate is still meaningful — a leaked label cannot cause a false positive on normal traffic. Closing that hole (label-independent address assignment, then re-measure both numbers) is open work; until then this row is a methodology note, not a result.

**Why the two "detection" rows disagree by three orders of magnitude**: they are different attacks with the same name. A concentrated flood trips a per-source connection limit; a distributed one is invisible to it by construction, and Kitsune scores per host, so spoofed sources each look like a well-behaved client. This is the boundary of the approach, not a tuning gap.

**Calibrate `blocking:` before trusting either number.** Strikes are counted per BLOCK verdict, i.e. per packet, so at a 2–3% packet-level false-positive rate a legitimate source sending a few hundred packets inside `strikes_window` reaches the shipped `strikes_threshold: 5` and takes a 10-minute kernel ban; repeated cycles persist a permanent one. Measure on your own traffic (`cli.py test --pcap`, watching `nips_alert_events_written_total`) and set the threshold to a multiple of what your traffic produces.

### Reproducing this on your own hardware

- `scripts/benchmark.py` — trains Kitsune on synthesized normal traffic, then reports rule-engine accuracy, training/detection throughput, and attack detection rate.
- `scripts/benchmark_nslkdd.py` — downloads NSL-KDD, maps flow records to synthetic packets, trains Kitsune on normal flows, and reports precision/recall/FPR.

Why detection on NSL-KDD is weak here: NSL-KDD records are **flow-level summaries**, not packet captures. Mapping each flow to a few packets throws away the timing and burst patterns that Kitsune learns from. Volumetric attacks (DoS, probe) survive the mapping better than content attacks (R2L, U2R), which look like ordinary TCP at the packet level. Treat the per-attack numbers as a statement of that limitation, not a measured accuracy claim.

The rule engine itself is exact: blacklist/whitelist, protocol filtering, and rate limiting are deterministic and always applied before the ML stage. Rate limit counts only TCP SYN (ACK clear) and UDP datagrams; established TCP sessions (ACK/data/FIN) do not consume the budget.

### Offline testing with your own traffic

The detection pipeline can be exercised **without** root or iptables, which is the practical way to measure what it would do on the traffic you actually carry:

- **Real pcap (the one that tells you something):** capture packets and run them through the pipeline offline.
  ```bash
  # capture 30s of live traffic (requires root for the sniff)
  sudo python -c "from scapy.all import sniff, wrpcap; wrpcap('cap.pcap', sniff(iface='en0', timeout=30))"
  # offline detection — no root needed
  python cli.py test --pcap cap.pcap
  ```
  This surfaces the **real** false-positive rate (e.g. legitimate ICMP being blocked by the protocol filter — in this offline path ICMP does reach the engine, unlike live interception with `intercept_icmp: false`), which the synthetic simulation does not. Note Kitsune needs ~55k normal packets before it leaves training mode, so short captures mostly exercise the rule engine.
- The synthetic simulation and the bundled reconstruction are covered by the table above; their limits are stated there rather than repeated here.

#### Fail-closed behavior

With learning detection **enabled**, a packet the rule engine does not decide must be scored by something. If every mounted detector that could score has raised or tripped its circuit breaker, the pipeline raises `DetectionUnavailable` and that packet is dropped rather than waved through: a silent outage is safer than allowing unknown traffic while the detector you paid for is dead. Turning `engine.ml.enabled` off is the opposite case — a decision, so undecided traffic is allowed, and it never triggers this path. A detector mounted without its model (`ready: false`) is treated as not deployed: it neither decides nor counts as coverage.

The status API exposes `detection_unavailable_drops`, `broken_detectors`, `ml_enabled`, `ml_consulted` and `ml_idle` so an operator can tell these three states apart instead of guessing from one boolean.

---

## Deployment

### Docker

```bash
# Build and start
bash deploy.sh build
bash deploy.sh start

# Verify tests pass (runs all verify_* scripts in container)
bash deploy.sh test

# View logs
bash deploy.sh logs

# Stop
bash deploy.sh stop
```

**Notes:**
- `deploy.sh` runs 8 verification scripts (`verify_engine_module`, `verify_interception_module`, `verify_block_lifecycle`, `verify_live_exposed_bugs`, `verify_fpr_regression`, `verify_features_module`, `verify_data_module`, `verify_management_plane`) instead of `pytest`.
- The container runs as non-root user `nips` for security. Bind-mount `rules.json` from the host — it must exist before `docker compose up` or you'll get an `IsADirectoryError`.
- **The container is the management plane only.** NFQUEUE and iptables need the host network stack, so the interceptor runs on the host (`cli.py start` or the systemd unit). Deploying only the container gives you an API that reports and edits rules but drops no traffic — a dashboard, not a prevention system.
- `rules.json` contains only **persistent** blacklist entries (operator-added + escalated permanent bans). Temp-ban mirrors live in the ephemeral tier and are never written to disk.

### Linux host

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python app.py

# Or use CLI directly (requires root for live interception)
sudo python cli.py start
```

### systemd

Two units under `deploy/systemd/`, meant to be installed as `nips-api.service` and `nips-interceptor.service` after editing the `/opt/nips` paths (use `systemctl edit` drop-ins rather than editing the shipped file):

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nips-api nips-interceptor
```

They are deliberately separate: only the interceptor holds the privileges that rewrite the firewall, and the web-facing process does not. The interceptor needs real root (`cli.py start` checks `geteuid()`), so capabilities alone will not satisfy it. Stop behaviour matters — the SIGTERM handler removes the NFQUEUE redirect, so `TimeoutStopSec` is generous; killing it faster leaves a kernel redirect with nothing draining the queue, which stalls traffic until the nfqueue timeout.

### Remote administration

The API listens on `api.host`/`api.port` and speaks plain HTTP. It has one shared token and no per-user identity, so it is designed to sit behind a TLS-terminating reverse proxy that also enforces source addresses or client certificates — not to be published directly. The shipped docker-compose binds it to `127.0.0.1:8000` for the same reason.

The CLI targets `http://127.0.0.1:8000` by default; point it elsewhere per invocation or by environment:

```bash
python cli.py --url https://nips.internal:8443 --token "$NIPS_API_TOKEN" status
NIPS_API_URL=http://10.0.0.5:8000 python cli.py alerts --last 20
```

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — layer design, data flow, module boundaries, red lines
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR workflow, pre-submission checklist, what we reject
- [CODE_STYLE.md](CODE_STYLE.md) — coding conventions, import rules, system call validation
- [SECURITY.md](SECURITY.md) — vulnerability reporting, deployment best practices
- [CHANGELOG.md](CHANGELOG.md) — release history
- API endpoints: see the "Run the API" section above (/docs, /redoc and OpenAPI are disabled in production)

---

## Contact

- **Author**: 梓铭
- **Email**: 2147514473@qq.com
- **Issues**: [GitHub Issues](https://github.com/zimingttkx/Network-Security-Based-On-ML/issues)

## License

MIT — see [LICENSE](LICENSE)
