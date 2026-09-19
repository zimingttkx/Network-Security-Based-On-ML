"""NIPS — Network Intrusion Prevention System.  FastAPI application."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, field_validator

from networksecurity.engine import DetectionPipeline, RuleEngine
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.observability import EventStore, configure_logging, render_metrics

# --- Application -----------------------------------------------------------

app = FastAPI(
    title="NIPS — Network Intrusion Prevention System",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# CORS + auth come from config (api block).  "*" origins are gone: this API
# can blacklist/whitelist arbitrary IPs and start/stop the interception
# engine, so an open-CORS management surface was a remote-DoS primitive.
from networksecurity.utils.config import (
    load_api_config,
    load_logging_config,
    load_storage_config,
)

# Log routing comes first: everything below (including the "auth is off"
# warning) should already reach the configured file/syslog sinks.  The
# "networksecurity." prefix is deliberate — configure_logging owns that
# subtree and leaves uvicorn's own logger alone.
logger = logging.getLogger("networksecurity.app")
configure_logging(load_logging_config())

_api_cfg = load_api_config()
API_AUTH_TOKEN: str = _api_cfg["auth_token"]

if not API_AUTH_TOKEN:
    logger.warning(
        "api.auth_token is EMPTY — the management API runs WITHOUT "
        "authentication.  Anyone who can reach this port can blacklist/"
        "whitelist arbitrary IPs and start/stop the engine.  Set "
        "api.auth_token in config.yaml or the NIPS_API_TOKEN env var."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_cfg["cors_origins"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_token(x_api_token: str = Header(default="")) -> None:
    """Dependency guarding every /api/v1/* route.

    /health stays open (liveness probes).  Comparison uses
    secrets.compare_digest: a plain == short-circuits, leaking the token
    prefix byte-by-byte through timing.
    """
    if not API_AUTH_TOKEN:
        return  # auth disabled by config (development mode)
    if not secrets.compare_digest(x_api_token, API_AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Token")

# --- Engine state -----------------------------------------------------------

RULES_FILE = Path(__file__).resolve().parent / "rules.json"

pipeline: DetectionPipeline = DetectionPipeline()

# Engine tuning comes from config/config.yaml (engine block) so operator
# overrides actually apply — previously only the interception block was read.
from networksecurity.utils.config import load_engine_config

_engine_cfg = load_engine_config()
pipeline.set_rule_engine(RuleEngine(
    window_seconds=_engine_cfg["rule_engine"]["window_seconds"],
    max_connections=_engine_cfg["rule_engine"]["max_connections"],
    allowed_protocols=set(_engine_cfg["rule_engine"]["allowed_protocols"]),
    allowed_icmp_types=set(_engine_cfg["rule_engine"]["allowed_icmp_types"]),
))
pipeline.add_detector(KitsuneDetector(
    max_autoencoder_size=_engine_cfg["kitsune"]["max_autoencoder_size"],
    threshold_percentile=_engine_cfg["kitsune"]["threshold_percentile"],
    learning_rate=_engine_cfg["kitsune"]["learning_rate"],
))
# Kitsune grace periods must be set before the first packet is processed.
for _d in pipeline.detectors:
    if isinstance(_d, KitsuneDetector):
        _d.set_grace_periods(
            fm_grace_period=_engine_cfg["kitsune"]["fm_grace_period"],
            ad_grace_period=_engine_cfg["kitsune"]["ad_grace_period"],
        )

# Optional: LUCID DDoS detector (requires TensorFlow).  It is added to the
# pipeline but stays inactive until a trained model is provided, so it does
# not silently no-op as an "active" detector.
try:
    from networksecurity.utils.config import load_lucid_config
    from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
    
    _lucid_cfg = load_lucid_config()
    _model_path = _lucid_cfg.get("model_path", "")
    
    if _model_path:
        _lucid_adapter = LucidDetectorAdapter(
            time_window=_lucid_cfg["time_window"],
            packets_per_flow=_lucid_cfg["packets_per_flow"],
            enabled=True,
        )
        
        # Load the model before registering the detector: a detector that
        # cannot load its weights must not join the pipeline at all.
        if asyncio.run(_lucid_adapter.load_model(_model_path)):
            pipeline.add_detector(_lucid_adapter)
        else:
            logger.warning("LUCID model at %r failed to load; detector not registered",
                           _model_path)
    else:
        pipeline.add_detector(LucidDetectorAdapter(enabled=False))
except ImportError:
    pass
except Exception:
    logger.exception("failed to initialize LUCID detector")

_interceptor: object | None = None  # Interceptor | None
_interceptor_thread: threading.Thread | None = None

# Alerts and the management audit trail live in SQLite, not in a list that a
# restart wipes: an operator investigating an incident at 09:00 needs the
# verdicts from 03:00.  Producers only enqueue (see EventStore) so the
# detection path never waits on the disk.
_storage_cfg = load_storage_config()
event_store = EventStore(
    _storage_cfg["events_db"],
    max_rows=_storage_cfg["max_rows"],
    retention_days=_storage_cfg["retention_days"],
    queue_size=_storage_cfg["queue_size"],
)

_start_lock: threading.Lock = threading.Lock()
start_time: datetime = datetime.now(tz=timezone.utc)


# --- Validation ------------------------------------------------------------

from networksecurity.utils.validation import (
    blacklist_refusal,
    sweep_refused_entries,
    validate_ip_or_cidr,
)
from networksecurity.engine.signature_engine import SignatureError
from networksecurity.utils.reload import ReloadProbe


def _record_alert(source_ip: str, reason: str, action: str, detector: str) -> None:
    event_store.record_alert(source_ip, reason, action, detector)


def _audit(request: Request, *, target: str, result: str, detail: str = "") -> None:
    """Record a management-plane action: who changed what, and whether it took.

    The actor is the peer address of the connection.  There is no per-user
    identity yet (the API has one shared token), so ``actor`` answers "which
    host", not "which person" — see the README's authentication note.
    """
    event_store.record_audit(
        actor=request.client.host if request.client else "unknown",
        method=request.method,
        path=request.url.path,
        target=target,
        result=result,
        detail=detail,
    )
    # Tell the middleware this request is already covered, so a handler-level
    # refusal (e.g. reload_failed on a 500) is not recorded a second time as a
    # bare status code.  The scope dict is the one object shared by both.
    request.scope["nips_audited"] = True


@app.middleware("http")
async def _audit_rejected_requests(request: Request, call_next):
    """Audit attempts the framework itself refuses.

    A malformed or unauthorised request never reaches a handler, so the
    handler-level ``_audit`` call cannot see it — yet a burst of 401s against
    the rule endpoints is exactly what an operator wants in the trail.
    """
    response = await call_next(request)
    if (request.url.path.startswith("/api/v1") and response.status_code >= 400
            and not request.scope.get("nips_audited")):
        event_store.record_audit(
            actor=request.client.host if request.client else "unknown",
            method=request.method, path=request.url.path, target="",
            result=str(response.status_code), detail="rejected before handler")
    return response


def _parse_time_bound(value: str | None) -> float | None:
    """Accept epoch seconds or an ISO8601 timestamp as a query filter."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=422,
                            detail=f"invalid timestamp {value!r}: use epoch seconds or ISO8601")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _blacklist_refusal(ip: str) -> str | None:
    """Why ``ip`` must not enter the blacklist, or ``None`` if it may."""
    from networksecurity.utils.config import load_interception_config
    safe_ips = load_interception_config().get("safe_ips") or []
    return blacklist_refusal(ip, safe_ips)


pipeline.rule_engine.load_rules(RULES_FILE)

# Drop entries the kernel would refuse anyway (loopback / safe_ips).  They
# can only be here via older versions or hand-edited rules.json; left in
# place they persist across restarts as phantom blocks.
_rules_swept = False
for _ip in sweep_refused_entries(pipeline.rule_engine):
    _rules_swept = True
if _rules_swept:
    pipeline.rule_engine.save_rules(RULES_FILE)

# Watches rules.json and config.yaml.  Passed to the interceptor so a
# hand-edited rule takes effect on the next sweep, and exposed through
# POST /api/v1/rules/reload for an explicit, audited apply.
reload_probe = ReloadProbe(pipeline.rule_engine, RULES_FILE)

# --- Pydantic models -------------------------------------------------------

class BlacklistEntry(BaseModel):
    ip: str
    reason: str = "manual"

    @field_validator("ip")
    @classmethod
    def _ip_ok(cls, v: str) -> str:
        v = validate_ip_or_cidr(v)
        # Loopback / safe-ips entries would be persisted (rules.json survives
        # restarts) while the kernel refuses to enforce them — a phantom
        # block.  Reject at the boundary instead of accepting a divergent
        # rule set.
        refusal = blacklist_refusal(v)
        if refusal is not None:
            raise ValueError(refusal)
        return v


class WhitelistEntry(BaseModel):
    ip: str

    @field_validator("ip")
    @classmethod
    def _ip_ok(cls, v: str) -> str:
        # Default routes are refused by validate_ip_or_cidr for both rule sets,
        # so whitelisting the entire internet cannot slip through here.
        return validate_ip_or_cidr(v)


# --- Health -----------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


# --- Status & stats --------------------------------------------------------

@app.get("/api/v1/status", dependencies=[Depends(require_token)])
async def engine_status():
    interceptor_running = (
        _interceptor is not None and getattr(_interceptor, "running", False)
    )
    # Kernel-level blocks: every iptables DROP the interceptor installed,
    # temp bans included.  Distinct from the rule engine's blacklist, which
    # reports the *persistent* tier only — a temp ban mirrors into the
    # ephemeral tier, so it shows up here and in /api/v1/blocks but not in
    # /api/v1/rules.
    kernel_blocked = (
        _interceptor.blocked_ips
        if _interceptor is not None and hasattr(_interceptor, "blocked_ips")
        else []
    )
    # Detection-loop health: seconds since the last packet completed
    # detection.  A hung loop fail-closes ALL traffic (fail-closed by
    # design), so surfacing staleness turns a silent network outage into a
    # visible alert.  None = interception never started.
    inter_status = (
        _interceptor.status()
        if _interceptor is not None and hasattr(_interceptor, "status")
        else {}
    )
    detect_stale = inter_status.get("detection_loop_stale_seconds")
    pipe_status = pipeline.status()
    status = {
        "running": interceptor_running or pipeline.running,
        "interception_active": interceptor_running,
        "uptime_seconds": (datetime.now(tz=timezone.utc) - start_time).total_seconds(),
        "detectors": pipe_status["detectors"],
        "broken_detectors": pipe_status["broken_detectors"],
        # degraded: some ML detectors are out.  ml_unavailable: all of them
        # are, so every packet the rule engine does not decide raises
        # DetectionUnavailable and is dropped — detection_unavailable_drops
        # counts those drops.
        "degraded": pipe_status["degraded"],
        "ml_unavailable": pipe_status["ml_unavailable"],
        "detection_unavailable_drops": inter_status.get(
            "detection_unavailable_drops"),
        # Packets the parser could not read completely and consistently, and
        # which were therefore dropped fail-closed.  A rising count here means
        # wire traffic is being discarded before detection ever sees it.
        "nfqueue_parse_failed": inter_status.get("nfqueue_parse_failed"),
        # False when ip6tables is missing: IPv6 bypasses the IPS entirely.
        "ipv6_intercepted": inter_status.get("ipv6_ready"),
        # Blocks the kernel refused and temp-ban lifts that failed.  Both are
        # retried by the sweeper, so a list that never drains means the
        # firewall and our view of it have diverged.
        "pending_enforce": inter_status.get("pending_enforce"),
        "pending_lift": inter_status.get("pending_lift"),
        "kitsune_trained": bool(
            hasattr(pipeline, "_detectors")
            and any(
                hasattr(d, "is_ready") and d.is_ready
                for d in getattr(pipeline, "_detectors", [])
            )
        ),
        "total_processed": pipeline.total_processed,
        "total_blocked": pipeline.total_blocked,
        "blocked_ips": pipeline.rule_engine.get_blacklist(),
        "kernel_blocked_ips": kernel_blocked,
        "detection_loop_stale_seconds": detect_stale,
        "last_reload": inter_status.get("last_reload"),
        # Event persistence health: a non-zero dropped/write_errors count
        # means the audit trail is incomplete, which an operator must be able
        # to see without reading logs.
        "event_store": event_store.stats(),
        # Reload counters: a rising "failures" means the operator's edits to
        # rules.json / config.yaml are being rejected and never applied.
        "reload": reload_probe.stats(),
    }
    return status


@app.get("/api/v1/stats/overview", dependencies=[Depends(require_token)])
async def stats_overview():
    return {
        "total_processed": pipeline.total_processed,
        "total_blocked": pipeline.total_blocked,
        "rule_engine": pipeline.rule_engine.stats(),
        "uptime_seconds": (datetime.now(tz=timezone.utc) - start_time).total_seconds(),
    }


# --- Alerts ----------------------------------------------------------------

_ALERT_FIELDS = ("timestamp", "source_ip", "reason", "action", "detector")
_AUDIT_FIELDS = ("timestamp", "actor", "method", "path", "target", "result", "detail")


def _serialize_export(items: list[dict], fields: tuple[str, ...], fmt: str) -> str:
    if fmt == "csv":
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
        return out.getvalue()
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


@app.get("/api/v1/alerts", dependencies=[Depends(require_token)])
async def get_alerts(request: Request, limit: int = 50, offset: int = 0,
                     source_ip: str | None = None, action: str | None = None,
                     since: str | None = None, until: str | None = None,
                     format: str = "json"):
    """Detection/block events, newest first, with optional filters.

    ``since``/``until`` take epoch seconds or ISO8601.  ``format=csv`` or
    ``jsonl`` returns a download for shipping to a SIEM; both stay bounded by
    ``limit`` (max 1000 rows per request) so an export cannot exhaust memory.
    """
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    page = event_store.query_alerts(
        limit=limit, offset=offset, source_ip=source_ip, action=action,
        since=_parse_time_bound(since), until=_parse_time_bound(until))
    if format in ("csv", "jsonl"):
        _audit(request, target=f"alerts?format={format}&limit={limit}",
               result="export", detail=f"total={page['total']}")
        media = "text/csv" if format == "csv" else "application/x-ndjson"
        return PlainTextResponse(
            _serialize_export(page["items"], _ALERT_FIELDS, format), media_type=media,
            headers={"Content-Disposition": f'attachment; filename="alerts.{format}"'})
    if format != "json":
        raise HTTPException(status_code=422, detail="format must be json, csv or jsonl")
    return page


@app.get("/api/v1/audit", dependencies=[Depends(require_token)])
async def get_audit(request: Request, limit: int = 50, offset: int = 0,
                    actor: str | None = None, result: str | None = None,
                    since: str | None = None, until: str | None = None,
                    format: str = "json"):
    """Who changed which rule or engine state, and whether it was accepted.

    ``actor`` is the peer address of the connection: the API has one shared
    token, so this identifies the host, not a named user.
    """
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    page = event_store.query_audit(
        limit=limit, offset=offset, actor=actor, result=result,
        since=_parse_time_bound(since), until=_parse_time_bound(until))
    if format in ("csv", "jsonl"):
        media = "text/csv" if format == "csv" else "application/x-ndjson"
        return PlainTextResponse(
            _serialize_export(page["items"], _AUDIT_FIELDS, format), media_type=media,
            headers={"Content-Disposition": f'attachment; filename="audit.{format}"'})
    if format != "json":
        raise HTTPException(status_code=422, detail="format must be json, csv or jsonl")
    return page


@app.get("/metrics", dependencies=[Depends(require_token)])
async def metrics():
    """Prometheus text exposition.  Guarded by the token like every other
    non-``/health`` route: the gauges describe live firewall policy."""
    return PlainTextResponse(
        render_metrics(pipeline=pipeline, store=event_store,
                       interceptor=_interceptor, started_at=start_time.timestamp()),
        media_type="text/plain; version=0.0.4; charset=utf-8")


# --- Rule management -------------------------------------------------------

@app.get("/api/v1/rules", dependencies=[Depends(require_token)])
async def get_rules():
    return {
        "blacklist": pipeline.rule_engine.get_blacklist(),
        "whitelist": pipeline.rule_engine.get_whitelist(),
    }


@app.get("/api/v1/blocks", dependencies=[Depends(require_token)])
async def get_blocks():
    """Escalation-policy view: who is observing / temp-banned / perm-banned.

    Complements /rules (the persisted operator blacklist) with the live
    strike counters and temp-ban TTLs so an operator can see WHY an IP is
    blocked and when a temp ban lifts.
    """
    if _interceptor is None or not hasattr(_interceptor, "_block_policy"):
        return {"items": []}
    return {"items": _interceptor._block_policy.snapshot()}


@app.post("/api/v1/rules/blacklist", dependencies=[Depends(require_token)])
async def add_blacklist(entry: BlacklistEntry, request: Request):
    pipeline.rule_engine.add_blacklist(entry.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    # Promote any temp-ban mirror for this IP to an operator entry, so the
    # expiry sweeper cannot remove the blacklist entry when the ban lifts.
    if _interceptor is not None and hasattr(_interceptor, "note_operator_blacklist"):
        try:
            _interceptor.note_operator_blacklist(entry.ip)
        except Exception:
            logger.exception("Failed to promote blacklist entry for %s", entry.ip)
    _record_alert(entry.ip, entry.reason, "block", "rule_engine")
    _audit(request, target=entry.ip, result="blacklist_add", detail=entry.reason)
    return {"status": "ok", "blacklist": pipeline.rule_engine.get_blacklist()}


@app.delete("/api/v1/rules/blacklist/{ip:path}", dependencies=[Depends(require_token)])
async def remove_blacklist(ip: str, request: Request):
    present = ip in pipeline.rule_engine.get_blacklist()
    pipeline.rule_engine.remove_blacklist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    # Keep the kernel-level enforcement in sync: if the interceptor
    # permanently blocked this IP via iptables (BLOCK verdict, mirrored into
    # the blacklist), lift that DROP too — otherwise the IP stays kernel-
    # banned while the rule engine reports it unblocked.
    unblocked = False
    if _interceptor is not None:
        try:
            unblocked = _interceptor.unblock_ip(ip)
        except Exception:
            logger.exception("Failed to lift kernel block for %s", ip)
    _audit(request, target=ip, result="blacklist_remove",
           detail=f"was_listed={present} kernel_block_removed={unblocked}")
    return {
        "status": "ok",
        "kernel_block_removed": unblocked,
        "blacklist": pipeline.rule_engine.get_blacklist(),
    }


@app.post("/api/v1/rules/whitelist", dependencies=[Depends(require_token)])
async def add_whitelist(entry: WhitelistEntry, request: Request):
    pipeline.rule_engine.add_whitelist(entry.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    _record_alert(entry.ip, "whitelist", "allow", "rule_engine")
    _audit(request, target=entry.ip, result="whitelist_add")
    return {"status": "ok", "whitelist": pipeline.rule_engine.get_whitelist()}


@app.delete("/api/v1/rules/whitelist/{ip:path}", dependencies=[Depends(require_token)])
async def remove_whitelist(ip: str, request: Request):
    present = ip in pipeline.rule_engine.get_whitelist()
    pipeline.rule_engine.remove_whitelist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    _audit(request, target=ip, result="whitelist_remove",
           detail=f"was_listed={present}")
    return {"status": "ok", "whitelist": pipeline.rule_engine.get_whitelist()}


@app.post("/api/v1/rules/reload", dependencies=[Depends(require_token)])
async def apply_reload(request: Request):
    """Re-read rules.json and the live engine knobs from config.yaml.

    Detection keeps running: the swap happens under the rule engine's lock, and
    a malformed file leaves the previous rule set in place and reports the
    error rather than applying half of it.  Kitsune's grace periods and
    threshold are deliberately not re-applied — see the summary.
    """
    summary = reload_probe.probe(force=True) or {}
    if summary.get("errors"):
        _audit(request, target="rules", result="reload_failed",
               detail="; ".join(summary["errors"])[:200])
        raise HTTPException(status_code=500, detail=summary["errors"])
    _audit(request, target="rules", result="reload",
           detail=json.dumps({k: summary[k] for k in summary if k != "errors"},
                             default=str)[:200])
    return {"status": "reloaded", **summary}


# --- Signature rules -------------------------------------------------------

@app.get("/api/v1/signatures", dependencies=[Depends(require_token)])
async def list_signatures():
    """Declared rules in evaluation order, plus their match counters."""
    return {"items": pipeline.rule_engine.signatures,
            "hits": pipeline.rule_engine.signature_hits()}


@app.post("/api/v1/signatures", dependencies=[Depends(require_token)])
async def create_signature(spec: dict, request: Request):
    """Add or edit a signature.  Re-posting an existing id replaces that rule.

    Validation lives in the engine (Signature.parse) rather than being
    duplicated in a request model, so the API, rules.json and hot reload all
    refuse exactly the same things — a rule with no matchers, a /0 source, a
    tcp_flags matcher on a non-TCP protocol.
    """
    try:
        stored = pipeline.rule_engine.add_signature(spec)
    except SignatureError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    pipeline.rule_engine.save_rules(RULES_FILE)
    _audit(request, target=stored["id"], result="signature_upsert",
           detail=json.dumps(stored, default=str)[:200])
    return {"status": "ok", "signature": stored,
            "signatures": pipeline.rule_engine.signatures}


@app.delete("/api/v1/signatures/{sid}", dependencies=[Depends(require_token)])
async def delete_signature(sid: str, request: Request):
    removed = pipeline.rule_engine.remove_signature(sid)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no signature with id {sid!r}")
    pipeline.rule_engine.save_rules(RULES_FILE)
    _audit(request, target=sid, result="signature_remove")
    return {"status": "ok", "signatures": pipeline.rule_engine.signatures}


# --- Engine control --------------------------------------------------------

@app.post("/api/v1/engine/start", dependencies=[Depends(require_token)])
async def engine_start(request: Request):
    """Start live interception (Linux, requires root)."""
    global _interceptor, _interceptor_thread

    if _interceptor is not None and getattr(_interceptor, "running", False):
        return {"status": "already_running"}

    # Thread-safe start: acquire lock, check again, spawn thread.  This
    # prevents a fast stop() -> start() cycle from having a stale thread rip
    # out the NEW interceptor's iptables rules (fail-open window).
    async def _start_locked():
        global _interceptor
        with _start_lock:
            if _interceptor is not None and getattr(_interceptor, "running", False):
                return {"status": "already_running"}
            
            # Validate environment synchronously before spawning background thread.
            import os
            import shutil

            if os.geteuid() != 0:
                raise HTTPException(
                    status_code=403,
                    detail="Live interception requires root privileges.",
                )
            if not shutil.which("iptables"):
                raise HTTPException(
                    status_code=400,
                    detail="iptables not found — Linux required for live interception.",
                )

            try:
                from networksecurity.interception import Interceptor
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="Interceptor unavailable — install NetfilterQueue on Linux",
                )

            # Load interception config so safe_ips / queue num from config.yaml are
            # actually applied (previously ignored; config.yaml was dead).
            from networksecurity.utils.config import load_interception_config, load_blocking_config
            inter_cfg = load_interception_config()
            blocking_cfg = load_blocking_config()

            from networksecurity.engine.block_policy import BlockPolicy
            policy = BlockPolicy(
                strikes_threshold=blocking_cfg["strikes_threshold"],
                strikes_window=blocking_cfg["strikes_window"],
                temp_ban_seconds=blocking_cfg["temp_ban_seconds"],
                temp_ban_count_to_perm=blocking_cfg["temp_ban_count_to_perm"],
                table_max=blocking_cfg["table_max"],
            )

            started = threading.Event()

            local_interceptor = Interceptor(
                pipeline,
                queue_num=inter_cfg.get("nfqueue_num", 0),
                safe_ips=inter_cfg.get("safe_ips"),
                on_verdict=lambda pkt, v: _record_alert(
                    pkt.src_ip, v.reason, v.action.value, v.detector
                ) if v.action.value == "block" else None,
                block_policy=policy,
                reload_probe=reload_probe.probe,
                intercept_icmp=inter_cfg.get("intercept_icmp", False),
            )

            # Reuse the Interceptor's own setup so the detection event loop is
            # created correctly (a missing loop would make _on_packet fail-closed and
            # drop every packet).  setup() installs iptables + creates the loop but
            # does NOT block on capture, so this handler can return promptly; a
            # background thread then drains the queue.
            try:
                local_interceptor.setup()
                started.set()
                _interceptor = local_interceptor
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to set up interception")
                raise HTTPException(status_code=500, detail=f"Setup failed: {e}")

            def _run(instance):
                try:
                    instance.begin_capture()
                except Exception:
                    logger.exception("Interceptor thread crashed")
                finally:
                    # Tear down ONLY the instance this thread owns.  Capturing the
                    # local reference (not the module-global) prevents a fast
                    # stop() -> start() cycle from having this stale thread rip out
                    # the NEW interceptor's iptables rules (fail-open window).
                    instance._running = False
                    instance._iptables.cleanup_all()

            _interceptor_thread = threading.Thread(
                target=_run, args=(local_interceptor,), daemon=True
            )
            _interceptor_thread.start()

            return {
                "status": "started" if started.is_set() else "start_pending",
                "pipeline": pipeline.status(),
            }

    result = await _start_locked()
    _audit(request, target="engine", result="start", detail=str(result.get("status")))
    return result


@app.post("/api/v1/engine/stop", dependencies=[Depends(require_token)])
async def engine_stop(request: Request):
    """Stop live interception and clean up iptables rules."""
    global _interceptor

    if _interceptor is None:
        return {"status": "not_running"}

    try:
        await _shutdown()
    except Exception:
        logger.exception("Error stopping interceptor")
    pipeline.rule_engine.save_rules(RULES_FILE)
    _audit(request, target="engine", result="stop")
    return {"status": "stopped"}


# --- Main ------------------------------------------------------------------

async def _shutdown():
    """Cleanup on shutdown: stop interceptor, clean up iptables."""
    global _interceptor
    if _interceptor is not None:
        try:
            _interceptor.stop()
        except Exception:
            logger.exception("Error stopping interceptor at shutdown")
        finally:
            _interceptor._iptables.cleanup_all()
            _interceptor = None


@app.on_event("shutdown")
async def fastapi_shutdown():
    await _shutdown()
    # Drain the queued events before exiting; a restart must not swallow the
    # last seconds of verdicts.
    event_store.close()


if __name__ == "__main__":
    # host/port come from config.yaml's api block; they used to be hardcoded
    # here, so the shipped config values were dead.
    uvicorn.run("app:app", host=_api_cfg["host"], port=_api_cfg["port"], reload=False)
