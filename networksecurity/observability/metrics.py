"""Prometheus text-exposition rendering for NIPS.

Stdlib only — the exposition format is a few lines of text, and pulling in
``prometheus_client`` would add a runtime dependency to a security control
plane for no benefit.  This module holds no state: every series is read from
the objects that already own the numbers (pipeline, event store, interceptor),
so there is no second copy to drift out of sync.
"""

from __future__ import annotations

import time

_HEADER = """# HELP nips_up Whether the NIPS management API is responding.
# TYPE nips_up gauge
# HELP nips_packets_processed_total Packets admitted into the detection pipeline.
# TYPE nips_packets_processed_total counter
# HELP nips_packets_blocked_total Packets carrying a BLOCK verdict.
# TYPE nips_packets_blocked_total counter
# HELP nips_detector_broken ML detectors the circuit breaker has taken out of service.
# TYPE nips_detector_broken gauge
# HELP nips_detection_unavailable_drops_total Packets dropped fail-closed because no ML detector ran.
# TYPE nips_detection_unavailable_drops_total counter
# HELP nips_nfqueue_parse_failed_total Frames the parser rejected and dropped fail-closed.
# TYPE nips_nfqueue_parse_failed_total counter
# HELP nips_ipv6_intercepted 1 when IPv6 is redirected and blockable; 0 means IPv6 bypasses the IPS entirely.
# TYPE nips_ipv6_intercepted gauge
# HELP nips_blacklist_size Persistent blacklist entries (rules.json backed).
# TYPE nips_blacklist_size gauge
# HELP nips_whitelist_size Whitelist entries.
# TYPE nips_whitelist_size gauge
# HELP nips_ephemeral_blacklist_size Temp-ban blacklist mirrors (never persisted).
# TYPE nips_ephemeral_blacklist_size gauge
# HELP nips_alert_events_dropped_total Events lost because the store queue was full.
# TYPE nips_alert_events_dropped_total counter
# HELP nips_event_store_degraded 1 when the event database is unusable and reads fall back to the ring buffer.
# TYPE nips_event_store_degraded gauge
# HELP nips_interception_active 1 when live interception is running.
# TYPE nips_interception_active gauge
# HELP nips_uptime_seconds Seconds since process start.
# TYPE nips_uptime_seconds gauge
"""


def _line(name: str, value, labels: dict | None = None) -> str:
    if labels:
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


def render_metrics(*, pipeline, store, interceptor, started_at: float) -> str:
    """Return the /metrics body.

    ``interceptor`` may be None (management API up, live interception not
    started); the corresponding series are then reported as zero rather than
    omitted, so a dashboard does not silently lose its panels.
    """
    status = pipeline.status()
    rules = pipeline.rule_engine.stats()
    now = time.time()
    inter_status = interceptor.status() if interceptor is not None and hasattr(
        interceptor, "status") else {}
    store_stats = store.stats()

    lines = [_HEADER.rstrip("\n")]
    lines.append(_line("nips_up", 1))
    lines.append(_line("nips_uptime_seconds", round(now - started_at, 3)))
    lines.append(_line("nips_packets_processed_total", status["total_processed"]))
    lines.append(_line("nips_packets_blocked_total", status["total_blocked"]))
    lines.append(_line("nips_interception_active",
                       1 if interceptor is not None and getattr(interceptor, "running", False) else 0))
    lines.append(_line("nips_detectors_total", len(status["detectors"])))
    for name in status["detectors"]:
        lines.append(_line("nips_detector_state", 1, {"detector": str(name), "state": "registered"}))
    for name in status["broken_detectors"]:
        lines.append(_line("nips_detector_broken", 1, {"detector": str(name)}))
    if not status["broken_detectors"]:
        lines.append(_line("nips_detector_broken", 0))
    lines.append(_line("nips_ml_unavailable", 1 if status["ml_unavailable"] else 0))
    lines.append(_line("nips_degraded", 1 if status["degraded"] else 0))
    lines.append(_line("nips_blacklist_size", rules["blacklist_size"]))
    lines.append(_line("nips_whitelist_size", rules["whitelist_size"]))
    lines.append(_line("nips_ephemeral_blacklist_size", rules["ephemeral_blacklist_size"]))

    drops = inter_status.get("detection_unavailable_drops")
    lines.append(_line("nips_detection_unavailable_drops_total", drops if drops is not None else 0))
    parse_failed = inter_status.get("nfqueue_parse_failed")
    lines.append(_line("nips_nfqueue_parse_failed_total", parse_failed if parse_failed is not None else 0))
    ipv6 = inter_status.get("ipv6_ready")
    lines.append(_line("nips_ipv6_intercepted", 1 if ipv6 else 0))
    stale = inter_status.get("detection_loop_stale_seconds")
    if stale is not None:
        lines.append(_line("nips_detection_loop_stale_seconds", round(float(stale), 3)))
    for key in ("pending_enforce", "pending_lift"):
        value = inter_status.get(key)
        if value is not None:
            lines.append(_line(f"nips_{key}", len(value) if isinstance(value, list) else value))

    lines.append(_line("nips_alert_events_dropped_total", store_stats["dropped"]))
    # Store-wide despite the name: `written` counts audit rows as well as alerts,
    # so reconciling this against GET /api/v1/alerts needs written_alerts, not
    # this series.  Renaming would break existing dashboards, so it stays.
    lines.append(_line("nips_alert_events_written_total", store_stats["written"]))
    lines.append(_line("nips_event_store_write_errors_total", store_stats["write_errors"]))
    lines.append(_line("nips_event_store_degraded", 1 if store_stats["degraded"] else 0))
    lines.append(_line("nips_event_store_pending", store_stats["pending"]))
    return "\n".join(lines) + "\n"
