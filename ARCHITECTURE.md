# NIPS Architecture

## Project Positioning

NIPS is a **server-side Network Intrusion Prevention System** for Linux.

- **Inbound only**: intercepts incoming traffic to the host. Does not inspect outbound traffic.
- **Kernel-level enforcement**: blocking happens via nfqueue inline drop and iptables DROP rules. No memory-flag-only "blocking".
- **Real traffic only**: every packet processed by the pipeline originates from the kernel netfilter subsystem via NFQUEUE. No synthetic traffic generation exists in the production code path.
- **Detection is a plug-in**: the rule engine is the permanent part of the chain; learning detectors are mounted from `config/config.yaml` and **none is mounted by default** (`engine.ml.enabled: false`). A deployment with `networksecurity/engine/kitsune/` and `networksecurity/engine/lucid/` deleted outright is a supported configuration, not a broken one — see "The Detector Contract".
- **Every claim has a check**: behaviours the docs assert are backed by runnable checks (`scripts/verify_*.py`, wired into CI), including the negative ones — what detection does *not* achieve is in README's "Measured results and limits" rather than hidden.

---

## Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│  LAYER 4 — Management Interface                      │
│  app.py (REST API)  +  cli.py (CLI)                  │
│  Responsibilities: status, alerts, rule CRUD, start/stop │
│  Constraints: read-only consumer of engine state.    │
│  Must NOT generate traffic or simulate alerts.       │
└────────────────────────┬─────────────────────────────┘
                         │  reads pipeline.status()
                         │  calls pipeline.process_packet() [test only]
┌────────────────────────┴─────────────────────────────┐
│  LAYER 3 — Detection Engine                          │
│  networksecurity/engine/                              │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │RuleEngine│→ │KitsuneDetector│→ │LucidDetector │  │
│  │ whitelist│  │ AfterImage    │  │ CNN DDoS     │  │
│  │ blacklist│  │ 90-dim stats │  │ flow detect  │  │
│  │ rate lim │  │ KitNET AE ens │  │              │  │
│  └──────────┘  └───────────────┘  └──────────────┘  │
│                         │                            │
│              Pipeline: short-circuit on BLOCK        │
│  Constraints: never calls iptables directly.         │
│  Must not generate traffic internally.              │
└────────────────────────┬─────────────────────────────┘
                         │  receives PacketInfo
┌────────────────────────┴─────────────────────────────┐
│  LAYER 2 — Feature Extraction                        │
│  networksecurity/features/                            │
│  AfterImage (90-dim incremental stats)               │
│  FlowTracker (5-tuple → flow features)               │
│  Constraints: features computed from packet fields.  │
│  No fixed templates, no random vectors.              │
└────────────────────────┬─────────────────────────────┘
                         │  receives raw packet bytes
┌────────────────────────┴─────────────────────────────┐
│  LAYER 1 — Traffic Interception (Linux only)         │
│  networksecurity/interception/                        │
│  NFQUEUE → PacketParser → Interceptor                │
│  iptables rule management                            │
│  Constraints: MUST have root, MUST have iptables,    │
│  MUST have NetfilterQueue.  Fails fast otherwise.    │
└──────────────────────────────────────────────────────┘
```

---

## Data Flow

```
NIC → iptables NFQUEUE target → nfqueue kernel queue
     → NFQueueHandler._handle_packet()
     → PacketParser.from_raw(bytes)
     → PacketInfo {src_ip, dst_ip, src_port, dst_port, protocol, packet_size, tcp_flags,
                ttl, icmp_type, icmp_code, ...}   # icmp_* are rule fields only, absent
                                                # from to_dict() to keep the 90-dim
                                                # AfterImage vector fixed
     → Interceptor._on_packet(packet_info) → bool
         → DetectionPipeline.process_packet(packet_info)
             → RuleEngine.process_packet()     # ordered, cheapest first:
                 # 1 whitelist  2 protocol/ICMP-type  3 blacklist (persistent +
                 # ephemeral)  4 signatures  5 rate limit.  A signature BLOCK is
                 # enforced inline like any other rule-engine verdict and does
                 # not feed BlockPolicy strikes.  Stage 2 admits ICMPv4 types
                 # from allowed_icmp_types and always admits the ICMPv6 types
                 # that maintain the link (2/133/134/135/136) — blocking those
                 # takes this host off the network rather than defending it.
             → KitsuneDetector.process_packet() # AfterImage → KitNET
             → LucidDetectorAdapter.process_packet()  # CNN flow detection
         → Verdict {action, confidence, reason}
     → if BLOCK:
         nf_packet.drop()            ← inline kernel drop (this packet never reaches app)
         if detector is an ML detector (rule-engine verdicts do NOT escalate):
             BlockPolicy.record_block()  ← strike counting; a single BLOCK installs NO kernel rule
                 → temp_banned: iptables DROP + in-memory blacklist mirror (TTL, auto-lifted;
                                only the mirror is removed — operator entries are never touched)
                 → perm_banned: iptables DROP + persisted blacklist entry (rules.json;
                                loaded back into the rule engine on restart, kernel DROP not reinstalled)
     → if ALLOW:
         nf_packet.accept()          ← packet delivered to application
```

---

## Module Boundaries

```
networksecurity/
  engine/           # Detection logic.  Pure Python, no OS calls.
    detector.py     # BaseDetector ABC, PacketInfo dataclass
    verdict.py      # Action, ThreatLevel, Verdict types
    rule_engine.py  # IP whitelist/blacklist, rate limiting, signature dispatch
    signature_engine.py  # Declarative rules: src/dst CIDR + protocol + ports +
                    # TCP flags + rate threshold; action block or log
    pipeline.py     # DetectionPipeline chain with short-circuit
    block_policy.py # BLOCK escalation policy: strikes → temp ban → permanent ban
    kitsune/        # AfterImage + KitNET anomaly detection (NDSS'18)
    lucid/          # CNN DDoS flow detection (IEEE TNSM 2020)

  interception/     # OS-level traffic capture and blocking (Linux only)
    nfqueue_handler.py  # NFQUEUE bind → raw bytes → callback(bool)
    packet_parser.py    # Raw IPv4 / IPv6(+ext headers) / TCP,UDP,ICMP binary → PacketInfo
    iptables.py         # iptables rule add/remove/cleanup
    interceptor.py      # Orchestrator: nfqueue + pipeline + iptables

  features/         # Statistical feature extraction from packets
    flow_extractor.py   # 5-tuple flow tracking → FlowFeatures
    feature_registry.py # Feature set names, dimensions, descriptions

  data/             # Offline data loading (dev/testing only)
    dataset_loader.py   # NSL-KDD, CICIDS2017, UNSW-NB15 labeled CSV/Parquet loader (header required)
    pcap_loader.py      # scapy pcap reader

  observability/    # Durable events and metrics.  Storage/logging only — no
                    # packet inspection, and no import of engine/ or
                    # interception/: callers hand it objects.
    alert_store.py  # SQLite (WAL) alerts + management audit trail, bounded
                    # non-blocking queue, batched writer thread, retention purge
    metrics.py      # Prometheus text exposition, rendered from live objects
    log_setup.py    # level / rotating file / syslog routing for this subtree

  utils/            # Shared configuration and input validation
    config.py       # config.yaml readers (engine / api / blocking / storage / logging blocks)
    validation.py   # IP/CIDR validation, blacklist refusal, rule sweep
    reload.py       # ReloadProbe: mtime watch over rules.json + config.yaml
```

### Dependency Rules

```
interception/ ──imports──→ engine/        ✓ allowed (Interceptor uses Pipeline)
interception/ ──imports──→ features/      ✓ allowed (optional)
engine/       ──imports──→ interception/  ✗ FORBIDDEN (engine must not call OS)
engine/signature_engine.py  standalone  ✓ (no OS calls, no imports outside engine)
engine/       ──imports──→ features/      ✓ allowed
app.py/cli.py ──imports──→ engine/        ✓ allowed
app.py/cli.py ──imports──→ interception/  ✓ allowed (lazy, only for start/stop)
app.py/cli.py ──imports──→ utils/         ✓ allowed (config loading)
app.py/cli.py ──imports──→ observability/  ✓ allowed (management plane owns persistence)
engine/       ──imports──→ observability/  ✗ FORBIDDEN (detection must not own disk I/O; verdicts reach the store through the caller's on_verdict callback)
interception/ ──imports──→ observability/  ✗ FORBIDDEN (same reason: a stalled disk write in the NFQUEUE callback fail-closes traffic)
observability/ ──imports──→ engine/        ✗ FORBIDDEN (receives duck-typed objects; stays testable alone)
features/     ──imports──→ engine/        ✓ allowed (uses PacketInfo)
data/         standalone                   ✓ (no internal deps)
```

---

## Red Lines

These rules are non-negotiable. Any code violating them will be rejected in PR review.

### Data Source

| Allowed | Forbidden |
|---------|-----------|
| NFQUEUE raw packet bytes | `np.random.randn()` feature vectors |
| pcap file (offline test) | `generate_fake_traffic()` |
| PacketParser.from_raw() | Hardcoded `PacketInfo(src_ip="10.0.0.1", ...)` |
| AfterImage.update_get_stats() from real fields | `return [0.1, 0.3, 0.5, ...]` fixed vector |

### Blocking Enforcement

| Allowed | Forbidden |
|---------|-----------|
| `nf_packet.drop()` | `packet.is_blocked = True` |
| `iptables -I NIPS -s IP -j DROP` | `self._blocked.add(ip)` as sole action |
| `subprocess.run(["iptables", ...], check=True)` | `print("Blocked IP")` with no kernel call |

### Code Quality

| Allowed | Forbidden |
|---------|-----------|
| Functions with callers | Orphan functions with zero call sites |
| `logger.warning("model not trained")` | Silent `return False` without explanation |
| Lazy imports for OS-specific modules | Hard imports that crash on unsupported platforms |

### Simulation Code Location

- **Only allowed in**: `scripts/` (benchmarks, evaluation, regression checks) — and `tests/` if one is ever added
- **Strictly forbidden in**: `networksecurity/engine/`, `networksecurity/interception/`, `networksecurity/features/`
- **Keywords scanned by CI**: `mock`, `simulate`, `fake`, `demo_data`, `generate_packet`, `test_traffic`
- **Rejected in review (not CI-scanned)**: `random.randint` / `np.random.randn` / `np.random.uniform` outside kitnet/autoencoder weight init

---

## Training Flow

Kitsune uses **online unsupervised learning** — no offline dataset required:

1. Deploy system on production host during normal traffic period
2. KitNET auto-trains over first ~55,000 packets (fm_grace + ad_grace)
3. After training, threshold set at 99th percentile of RMSE
4. System transitions to detection mode automatically

LUCID requires **offline supervised training** on labeled DDoS datasets:

1. Prepare labeled flow data (CICIDS2017 DDoS subset or similar)
2. **Preprocess the dataset yourself first** — `DatasetLoader` assumes a **header-bearing CSV** (or an equivalent `.parquet` file) with the dataset's standard column names. It does **not** detect, convert, or add headers, and does **not** handle the raw headerless NSL-KDD `.txt` distribution (add the 41 standard feature names + `difficulty` + `label`). Preprocessing is the operator's responsibility; the loader only reads the prepared file. See *Training dataset preparation* in README.
3. Train CNN with `LucidDetector.train(X, y)`
4. Save model with `LucidDetector.save(path)`
5. Load model with `LucidDetector.load(path)` before deployment

---

## The Detector Contract

A detector implements `BaseDetector` (`networksecurity/engine/detector.py`). That is the entire surface a third-party module has to satisfy — nothing in `app.py` or `cli.py` changes to add one.

| member | contract |
|---|---|
| `async process_packet(packet: PacketInfo) -> Verdict \| None` | `None` abstains and hands the packet to the next detector. `BLOCK` ends the chain and the packet is dropped; any other explicit verdict is equally final — it ends the chain and is what gets enforced. |
| `configure(params: dict) -> None` | Called with the entry's `params:` before the first packet. Reject keys you do not understand: a silently ignored option is indistinguishable from a configured detector. |
| `ready -> bool` | `False` means "cannot score at all" (no model loaded, for example). Such a detector is consulted on no packet **and does not count as ML coverage** — see the table below. Warm-up is not `False`: a detector that is still training is covering traffic, and calling that an outage would drop every packet at startup. Whether it is emitting verdicts *yet* belongs in `status()`. |
| `status() -> dict` | Collected per detector into `detector_status` on `/api/v1/status`, and merged into the pipeline snapshot by name. Publish what an operator would need in order to notice you stopped working (Kitsune reports `trained`). Called outside the pipeline's status lock, and an exception inside it is contained to that detector's entry — a third-party `status()` must not be able to take the endpoint down. |

Mount it from config:

```yaml
engine:
  ml:
    enabled: true
    detectors:
      - uses: networksecurity.engine.threshold_detector:ThresholdDetector
        params: {window_seconds: 5, max_packets: 1000}
```

`uses` is a built-in short name (`kitsune`, `lucid` — those read their tuning from the `engine.kitsune` / `engine.lucid` blocks) or a `package.module:ClassName` path. A detector that cannot be constructed or configured is logged and skipped: one bad entry must not stop the management plane from starting.

`networksecurity/engine/threshold_detector.py` is the worked example — small, deterministic, and complete enough to copy.

### Fail-closed follows the switch, not luck

| state | a packet the rules did not decide |
|---|---|
| `engine.ml.enabled: false` | **ALLOW** — running without learning detection is a decision, not a failure |
| ML on, and **no** `ready` detector could execute (all raised, or all tripped) | **DROP** — an outage must not quietly become an open port |
| ML on, one detector dead but another ready one abstained | **ALLOW** — the surviving detector is the coverage; a partial outage is not a total one |
| ML on, but only `ready: false` detectors were mounted | **ALLOW**, loudly: the startup log says no detector was mounted |

Two of these rows used to be one, and wrong. An adapter with no model answered `LOG` rather than abstaining, which both ended the chain and counted as coverage — so a tripped live detector behind it switched fail-closed off silently while the status page still listed three detectors. `ready` exists to keep "not deployed", "could not run" and "running" from collapsing into the same sentence.

### Do NOT

- Add a "test mode" branch inside a detector that returns invented results
- Generate packets inside a detector — offline traffic belongs in `scripts/`
- Call iptables or OS commands from a detector; that is `interception/`'s job
- Read `config.yaml` from inside a detector; tuning arrives through `configure()`
