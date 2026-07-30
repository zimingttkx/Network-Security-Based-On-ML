"""NIPS — Network Intrusion Prevention System.  FastAPI application."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass

templates = Jinja2Templates(directory="templates")

# --- Engine state -----------------------------------------------------------

RULES_FILE = Path("rules.json")

pipeline: DetectionPipeline = DetectionPipeline()
pipeline.add_detector(KitsuneDetector())

# Optional: LUCID DDoS detector (requires TensorFlow)
try:
    from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
    pipeline.add_detector(LucidDetectorAdapter())
except ImportError:
    pass

_interceptor: Optional[object] = None  # Interceptor | None
_interceptor_thread: Optional[threading.Thread] = None

alerts: list[dict] = []
start_time: datetime = datetime.now()


def _load_rules() -> None:
    if not RULES_FILE.exists():
        return
    try:
        data = json.loads(RULES_FILE.read_text())
        for ip in data.get("blacklist", []):
            pipeline.rule_engine.add_blacklist(ip)
        for ip in data.get("whitelist", []):
            pipeline.rule_engine.add_whitelist(ip)
    except Exception:
        pass


def _save_rules() -> None:
    data = {
        "blacklist": pipeline.rule_engine.get_blacklist(),
        "whitelist": pipeline.rule_engine.get_whitelist(),
    }
    RULES_FILE.write_text(json.dumps(data, indent=2))


def _record_alert(source_ip: str, reason: str, action: str, detector: str) -> None:
    alerts.insert(0, {
        "timestamp": datetime.now().isoformat(),
        "source_ip": source_ip,
        "reason": reason,
        "action": action,
        "detector": detector,
    })
    if len(alerts) > 500:
        alerts.pop()


_load_rules()

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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- Status & stats --------------------------------------------------------

@app.get("/api/v1/status")
async def engine_status():
    interceptor_running = (
        _interceptor is not None and getattr(_interceptor, "running", False)
    )
    status = {
        "running": interceptor_running or pipeline.running,
        "interception_active": interceptor_running,
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
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
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
    }


# --- Alerts ----------------------------------------------------------------

@app.get("/api/v1/alerts")
async def get_alerts(limit: int = 50, offset: int = 0):
    return {
        "total": len(alerts),
        "items": alerts[offset : offset + limit],
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
    _save_rules()
    _record_alert(entry.ip, entry.reason, "block", "rule_engine")
    return {"status": "ok", "blacklist": pipeline.rule_engine.get_blacklist()}


@app.delete("/api/v1/rules/blacklist/{ip}")
async def remove_blacklist(ip: str):
    pipeline.rule_engine.remove_blacklist(ip)
    _save_rules()
    return {"status": "ok", "blacklist": pipeline.rule_engine.get_blacklist()}


@app.post("/api/v1/rules/whitelist")
async def add_whitelist(entry: WhitelistEntry):
    pipeline.rule_engine.add_whitelist(entry.ip)
    _save_rules()
    return {"status": "ok", "whitelist": pipeline.rule_engine.get_whitelist()}


@app.delete("/api/v1/rules/whitelist/{ip}")
async def remove_whitelist(ip: str):
    pipeline.rule_engine.remove_whitelist(ip)
    _save_rules()
    return {"status": "ok", "whitelist": pipeline.rule_engine.get_whitelist()}


# --- Engine control --------------------------------------------------------

@app.post("/api/v1/engine/start")
async def engine_start():
    """Start live interception (Linux, requires root)."""
    global _interceptor, _interceptor_thread

    if _interceptor is not None and getattr(_interceptor, "running", False):
        return {"status": "already_running"}

    # Validate environment synchronously before spawning background thread.
    import os as _os, shutil as _shutil

    if _os.geteuid() != 0:
        raise HTTPException(
            status_code=403,
            detail="Live interception requires root privileges.",
        )
    if not _shutil.which("iptables"):
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

    _interceptor = Interceptor(pipeline)

    def _run():
        try:
            _interceptor.start()
        except Exception:
            pass

    _interceptor_thread = threading.Thread(target=_run, daemon=True)
    _interceptor_thread.start()

    # Wait briefly to confirm startup
    await asyncio.sleep(0.5)
    running = getattr(_interceptor, "running", False)
    return {
        "status": "started" if running else "start_pending",
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
        pass
    _save_rules()
    _interceptor = None
    return {"status": "stopped"}


# --- Main ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
