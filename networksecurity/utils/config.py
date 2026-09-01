"""Config loading for NIPS.

Thin wrapper around the repo's config/config.yaml.  The interception layer
reads its ``safe_ips`` / ``nfqueue_num`` from here so that operator-tuned
values in the YAML file are actually applied at runtime instead of silently
ignored.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Resolve relative to the package root so the loader works regardless of the process CWD (e.g. when launched as a systemd service).
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

# Fallback so the system still starts if config.yaml is missing or malformed.
# Keep the IPv6 loopback in the fallback too, so a degraded config still
# protects ::1 (otherwise the ip6tables safe-ip path in IptablesManager
# would never receive ::1 and loopback IPv6 traffic could be intercepted).
_DEFAULT_INTERCEPTION = {
    "nfqueue_num": 0,
    "safe_ips": ["127.0.0.1", "::1"],
}


def load_interception_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``interception`` block of config.yaml.

    Returns a safe default if the file is absent or cannot be parsed, so a
    misconfigured host degrades to the conservative default rather than
    crashing the API at startup.
    """
    path = Path(path)
    if not path.exists():
        return dict(_DEFAULT_INTERCEPTION)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        # Never let a bad config file take down interception startup.
        return dict(_DEFAULT_INTERCEPTION)
    inter = data.get("interception") or {}
    return {
        "nfqueue_num": inter.get("nfqueue_num", _DEFAULT_INTERCEPTION["nfqueue_num"]),
        "safe_ips": inter.get("safe_ips", list(_DEFAULT_INTERCEPTION["safe_ips"])),
    }


# Defaults mirror the values documented in config/config.yaml so a missing
# file behaves exactly like the shipped configuration.
_DEFAULT_ENGINE = {
    "kitsune": {
        "max_autoencoder_size": 10,
        "fm_grace_period": 5000,
        "ad_grace_period": 50000,
        "learning_rate": 0.1,
        "threshold_percentile": 99.0,
    },
    "rule_engine": {
        "window_seconds": 1.0,
        "max_connections_per_window": 100,
        "allowed_protocols": [6, 17],
    },
}


def load_engine_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``engine`` block of config.yaml with per-section defaults.

    ``config.yaml``'s engine settings were previously dead (only the
    ``interception`` block was ever read), so operator-tuned grace periods and
    the rule-engine rate cap were silently ignored.  This loader makes the
    documented "config drives the engine" contract real.  Same degradation
    policy as ``load_interception_config``: a missing/malformed file falls
    back to the shipped defaults instead of failing startup.
    """
    path = Path(path)
    data: dict = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            # Never let a bad config file take down engine startup.
            data = {}
    engine = data.get("engine") or {}

    kitsune_cfg = engine.get("kitsune") or {}
    kitsune = {
        key: kitsune_cfg.get(key, default)
        for key, default in _DEFAULT_ENGINE["kitsune"].items()
    }

    re_cfg = engine.get("rule_engine") or {}
    rate_cfg = re_cfg.get("rate_limit") or {}
    rule_engine = {
        "window_seconds": rate_cfg.get("window_seconds",
                                       _DEFAULT_ENGINE["rule_engine"]["window_seconds"]),
        "max_connections": rate_cfg.get(
            "max_connections_per_window",
            _DEFAULT_ENGINE["rule_engine"]["max_connections_per_window"]),
        "allowed_protocols": re_cfg.get(
            "allowed_protocols",
            _DEFAULT_ENGINE["rule_engine"]["allowed_protocols"]),
    }
    return {"kitsune": kitsune, "rule_engine": rule_engine}
