"""Observability layer: durable events, metrics exposition and log routing.

Depended on only by the management plane (app.py / cli.py).  It must never be
imported by engine/ or interception/, mirroring the rule that the detection
path cannot reach the OS: the packet path hands events to this layer through
callbacks and never calls into it directly.
"""

from networksecurity.observability.alert_store import EventStore
from networksecurity.observability.log_setup import configure_logging
from networksecurity.observability.metrics import render_metrics

__all__ = ["EventStore", "configure_logging", "render_metrics"]
