# NIPS Architecture

## Project Positioning

NIPS is a **server-side Network Intrusion Prevention System** for Linux.

- **Inbound only**: intercepts incoming traffic to the host. Does not inspect outbound traffic.
- **Kernel-level enforcement**: blocking happens via nfqueue inline drop and iptables DROP rules. No memory-flag-only "blocking".
- **Real traffic only**: every packet processed by the pipeline originates from the kernel netfilter subsystem via NFQUEUE. No synthetic traffic generation exists in the production code path.

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

## Adding a New Detector

1. Create a new module in `networksecurity/engine/<name>/`
2. Implement `async def process_packet(self, packet: PacketInfo) -> Verdict | None`
3. Wire it into `DetectionPipeline` via `pipeline.add_detector()`
4. Add an adapter if the underlying algorithm has a different interface (see `detector_adapter.py` in kitsune/lucid)

Do NOT:
- Add a separate "test mode" path in the detector that returns fake results
- Generate synthetic packets inside the detector
- Call iptables or OS commands from inside the detector (that belongs in interception/)
