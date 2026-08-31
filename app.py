"""NIPS — Network Intrusion Prevention System.  FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from networksecurity.engine import DetectionPipeline
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector

# --- Application -----------------------------------------------------------

app = FastAPI(
    title="NIPS — Network Intrusion Prevention System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass

templates = Jinja2Templates(directory="templates")

# --- Engine state -----------------------------------------------------------

RULES_FILE = Path(__file__).resolve().parent / "rules.json"

logger = logging.getLogger(__name__)

pipeline: DetectionPipeline = DetectionPipeline()
pipeline.add_detector(KitsuneDetector())

# Optional: LUCID DDoS detector (requires TensorFlow).  It is added to the
# pipeline but stays inactive until a trained model is provided, so it does
# not silently no-op as an "active" detector.
try:
    from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
    pipeline.add_detector(LucidDetectorAdapter(enabled=False))
except ImportError:
    pass

_interceptor: object | None = None  # Interceptor | None
_interceptor_thread: threading.Thread | None = None

alerts: list[dict] = []
_alerts_lock: threading.Lock = threading.Lock()
start_time: datetime = datetime.now(tz=timezone.utc)


def _record_alert(source_ip: str, reason: str, action: str, detector: str) -> None:
    with _alerts_lock:
        alerts.insert(0, {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "source_ip": source_ip,
            "reason": reason,
            "action": action,
            "detector": detector,
        })
        if len(alerts) > 500:
            alerts.pop()


pipeline.rule_engine.load_rules(RULES_FILE)

# --- Pydantic models -------------------------------------------------------

class BlacklistEntry(BaseModel):
    ip: str
    reason: str = "manual"


class WhitelistEntry(BaseModel):
    ip: str


# --- Page routes -----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"page": "home"})


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


# --- Status & stats --------------------------------------------------------

@app.get("/api/v1/status")
async def engine_status():
    interceptor_running = (
        _interceptor is not None and getattr(_interceptor, "running", False)
    )
    status = {
        "running": interceptor_running or pipeline.running,
        "interception_active": interceptor_running,
        "uptime_seconds": (datetime.now(tz=timezone.utc) - start_time).total_seconds(),
        "detectors": pipeline.status()["detectors"],
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
    }
    return status


@app.get("/api/v1/stats/overview")
async def stats_overview():
    return {
        "total_processed": pipeline.total_processed,
        "total_blocked": pipeline.total_blocked,
        "rule_engine": pipeline.rule_engine.stats(),
        "uptime_seconds": (datetime.now(tz=timezone.utc) - start_time).total_seconds(),
    }


# --- Alerts ----------------------------------------------------------------

@app.get("/api/v1/alerts")
async def get_alerts(limit: int = 50, offset: int = 0):
    with _alerts_lock:
        return {
            "total": len(alerts),
            "items": list(alerts[offset : offset + limit]),
        }


# --- Rule management -------------------------------------------------------

@app.get("/api/v1/rules")
async def get_rules():
    return {
        "blacklist": pipeline.rule_engine.get_blacklist(),
        "whitelist": pipeline.rule_engine.get_whitelist(),
    }


@app.post("/api/v1/rules/blacklist")
async def add_blacklist(entry: BlacklistEntry):
    pipeline.rule_engine.add_blacklist(entry.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    _record_alert(entry.ip, entry.reason, "block", "rule_engine")
    return {"status": "ok", "blacklist": pipeline.rule_engine.get_blacklist()}


@app.delete("/api/v1/rules/blacklist/{ip}")
async def remove_blacklist(ip: str):
    pipeline.rule_engine.remove_blacklist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    return {"status": "ok", "blacklist": pipeline.rule_engine.get_blacklist()}


@app.post("/api/v1/rules/whitelist")
async def add_whitelist(entry: WhitelistEntry):
    pipeline.rule_engine.add_whitelist(entry.ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    return {"status": "ok", "whitelist": pipeline.rule_engine.get_whitelist()}


@app.delete("/api/v1/rules/whitelist/{ip}")
async def remove_whitelist(ip: str):
    pipeline.rule_engine.remove_whitelist(ip)
    pipeline.rule_engine.save_rules(RULES_FILE)
    return {"status": "ok", "whitelist": pipeline.rule_engine.get_whitelist()}


# --- Engine control --------------------------------------------------------

@app.post("/api/v1/engine/start")
async def engine_start():
    """Start live interception (Linux, requires root)."""
    global _interceptor, _interceptor_thread

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
    from networksecurity.utils.config import load_interception_config
    inter_cfg = load_interception_config()

    started = threading.Event()

    local_interceptor = Interceptor(
        pipeline,
        queue_num=inter_cfg.get("nfqueue_num", 0),
        safe_ips=inter_cfg.get("safe_ips"),
        on_verdict=lambda pkt, v: _record_alert(
            pkt.src_ip, v.reason, v.action.value, v.detector
        ),
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


@app.post("/api/v1/engine/stop")
async def engine_stop():
    """Stop live interception and clean up iptables rules."""
    global _interceptor

    if _interceptor is None:
        return {"status": "not_running"}

    try:
        _interceptor.stop()
    except Exception:
        logger.exception("Error stopping interceptor")
    pipeline.rule_engine.save_rules(RULES_FILE)
    _interceptor = None
    return {"status": "stopped"}


# --- Main ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
