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
