"""Mount the learning detectors that config asks for — and nothing else.

Both entrypoints (``app.py`` for the API, ``cli.py`` for live interception) call
:func:`attach_detectors`, so the two paths cannot drift into different chains.

The point of keeping this out of the entrypoints is that with
``engine.ml.enabled: false`` **no ML module is imported at all**: the built-in
detectors are imported inside the functions that build them.  A deployment that
runs on rules alone must keep working if ``networksecurity/engine/kitsune/`` is
deleted outright, and an import at module top would break that promise.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from networksecurity.engine.detector import BaseDetector
    from networksecurity.engine.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)

# Short names reserved for detectors shipped with the project.  Anything else
# in `uses:` is an import path, so a third-party module is addressed the same
# way a built-in is.
BUILTIN_NAMES = frozenset({"kitsune", "lucid"})


def _load_class(uses: str) -> type:
    """Resolve `module.path:Class` to a class."""
    module_name, sep, class_name = uses.partition(":")
    if not sep or not class_name:
        raise ValueError(f"detector {uses!r} is not 'module.path:ClassName'")
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"cannot load detector {uses!r}: {exc}") from exc


def _build_kitsune(engine_cfg: dict) -> BaseDetector:
    from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector

    k = engine_cfg["kitsune"]
    det = KitsuneDetector(
        max_autoencoder_size=k["max_autoencoder_size"],
        threshold_percentile=k["threshold_percentile"],
        learning_rate=k["learning_rate"],
    )
    # Grace periods must be set before the first packet is processed.
    det.set_grace_periods(fm_grace_period=k["fm_grace_period"],
                          ad_grace_period=k["ad_grace_period"])
    return det


def _build_lucid(engine_cfg: dict) -> BaseDetector | None:
    """Build LUCID only if a trained model exists and loads.

    Returns None (and says so) rather than mounting an adapter that cannot
    score: a registered no-op reads as coverage on the status page.
    """
    import asyncio

    from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
    from networksecurity.utils.config import load_lucid_config

    lc = load_lucid_config()
    if not lc.get("model_path"):
        logger.info("engine.lucid.model_path is empty — LUCID is not deployed "
                    "and no adapter is mounted")
        return None
    det = LucidDetectorAdapter(time_window=lc["time_window"],
                              packets_per_flow=lc["packets_per_flow"],
                              enabled=True)
    if not asyncio.run(det.load_model(lc["model_path"])):
        logger.warning("LUCID model at %r failed to load — detector not mounted",
                       lc["model_path"])
        return None
    return det


_BUILDS = {"kitsune": _build_kitsune, "lucid": _build_lucid}


def attach_detectors(pipeline: DetectionPipeline, ml_cfg: dict,
                     engine_cfg: dict) -> list[str]:
    """Mount every enabled detector onto ``pipeline``; return the mounted names.

    A detector that fails to build or configure is reported and skipped: one
    bad entry must not take the management plane down, and the operator learns
    about it from ``ml_idle`` / the log rather than from a silent gap.
    """
    mounted: list[str] = []
    if not ml_cfg["enabled"]:
        logger.info("engine.ml.enabled=false — running on the rule engine alone; "
                    "no learning detector is imported or constructed")
        return mounted

    for entry in ml_cfg["detectors"]:
        uses, params = entry["uses"], entry.get("params") or {}
        if not entry.get("enabled", True):
            logger.info("detector %r is present in config but enabled=false — "
                        "not mounted", uses)
            continue
        try:
            if uses in _BUILDS:
                det = _BUILDS[uses](engine_cfg)
            else:
                det = _load_class(uses)()
            if det is None:
                continue
            det.configure(params)
        except Exception:
            logger.exception("detector %r could not be mounted — skipped", uses)
            continue
        pipeline.add_detector(det)
        mounted.append(det.name)
        logger.info("mounted detector %r (params=%r)", det.name, params or {})

    if not mounted:
        logger.warning("engine.ml.enabled=true but no detector mounted — this is "
                       "rules-only operation with the switch left on; every "
                       "entry above was skipped or returned nothing")
    return mounted
