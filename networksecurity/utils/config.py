"""Config loading for NIPS.

Thin wrapper around the repo's config/config.yaml.  Every loader validates
types and ranges per key: an invalid or empty value falls back to the shipped
default with a WARNING instead of poisoning the runtime (an unvalidated
``max_connections_per_window: null`` used to raise TypeError inside the rate
limiter on every packet until the circuit breaker skipped RuleEngine for
good — a silent, system-wide fail-open).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Resolve relative to the package root so the loader works regardless of the process CWD (e.g. when launched as a systemd service).
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


def _load_mapping(path: str | Path) -> dict:
    """Read *path* and return its top-level mapping, or {} on any problem.

    Covers the three degradation cases uniformly: file missing, YAML parse
    error, and a valid YAML document whose top level is not a mapping (a bare
    list/scalar used to raise AttributeError out of every loader, taking the
    API down at import time).
    """
    path = Path(path)
    if not path.exists():
        logger.warning("config file %s not found; falling back to defaults", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        logger.warning("config file %s unreadable (%s); falling back to defaults", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "config %s top level is %s, expected a mapping; falling back to defaults",
            path, type(data).__name__,
        )
        return {}
    return data


def _as_mapping(data: dict, key: str) -> dict:
    """Return ``data[key]`` when it is a dict, else {} (with a warning)."""
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        logger.warning("config key %r is %s, expected a mapping; using defaults",
                       key, type(value).__name__)
        return {}
    return value


def _valid(value, default, cast, *, lo=None, hi=None, name: str = "value"):
    """Validate one scalar: cast + optional inclusive bounds, else *default*.

    bool is rejected for numeric casts because YAML ``true``/``false`` would
    otherwise silently become 1/0.
    """
    if value is None:
        logger.warning("config %s is empty; using default %r", name, default)
        return default
    if isinstance(value, bool):
        logger.warning("config %s is a boolean; using default %r", name, default)
        return default
    try:
        out = cast(value)
    except (TypeError, ValueError):
        logger.warning("config %s=%r is not a valid %s; using default %r",
                       name, value, cast.__name__, default)
        return default
    if lo is not None and out < lo:
        logger.warning("config %s=%r below minimum %r; using default %r", name, out, lo, default)
        return default
    if hi is not None and out > hi:
        logger.warning("config %s=%r above maximum %r; using default %r", name, out, hi, default)
        return default
    return out


# Fallback so the system still starts if config.yaml is missing or malformed.
# Keep the IPv6 loopback in the fallback too, so a degraded config still
# protects ::1 (otherwise the ip6tables safe-ip path in IptablesManager
# would never receive ::1 and loopback IPv6 traffic could be intercepted).
_DEFAULT_INTERCEPTION = {
    "nfqueue_num": 0,
    "safe_ips": ["127.0.0.1", "::1"],
    # Off by default: redirecting ICMP into NFQUEUE makes every host on the
    # segment wait on userspace verdicts for ping/PMTUD, and a stalled detection
    # loop would drop them fail-closed.
    "intercept_icmp": False,
}


def load_interception_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``interception`` block of config.yaml.

    Returns a safe default if the file is absent or cannot be parsed, so a
    misconfigured host degrades to the conservative default rather than
    crashing the API at startup.  ``safe_ips`` must be a list of strings;
    anything else (a bare string used to be iterated character by character,
    silently dropping loopback protection) falls back to the default.
    """
    data = _load_mapping(path)
    inter = _as_mapping(data, "interception")

    nfqueue_num = _valid(inter.get("nfqueue_num", _DEFAULT_INTERCEPTION["nfqueue_num"]),
                         _DEFAULT_INTERCEPTION["nfqueue_num"], int,
                         lo=0, hi=65535, name="interception.nfqueue_num")

    raw_safe = inter.get("safe_ips")
    if raw_safe is None:
        safe_ips = list(_DEFAULT_INTERCEPTION["safe_ips"])
    elif isinstance(raw_safe, list) and all(isinstance(x, str) for x in raw_safe):
        safe_ips = list(raw_safe)
    else:
        logger.warning("interception.safe_ips=%r is not a list of strings; "
                       "using default %r", raw_safe, _DEFAULT_INTERCEPTION["safe_ips"])
        safe_ips = list(_DEFAULT_INTERCEPTION["safe_ips"])

    raw_icmp = inter.get("intercept_icmp", _DEFAULT_INTERCEPTION["intercept_icmp"])
    if not isinstance(raw_icmp, bool):
        logger.warning("interception.intercept_icmp=%r is not a boolean; using default",
                       raw_icmp)
        raw_icmp = _DEFAULT_INTERCEPTION["intercept_icmp"]

    return {"nfqueue_num": nfqueue_num, "safe_ips": safe_ips, "intercept_icmp": raw_icmp}


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
        "allowed_icmp_types": [],
    },
}


def load_engine_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``engine`` block of config.yaml with per-section defaults.

    Every value is type/range-checked; an invalid entry logs a warning and
    falls back to the shipped default.  This is load-bearing for fail-closed
    behaviour: unchecked values (e.g. an empty ``max_connections_per_window``)
    used to raise inside per-packet code paths until the detector circuit
    breaker permanently skipped RuleEngine.
    """
    data = _load_mapping(path)
    engine = _as_mapping(data, "engine")
    kitsune_cfg = _as_mapping(engine, "kitsune")
    kd = _DEFAULT_ENGINE["kitsune"]

    kitsune = {
        "max_autoencoder_size": _valid(kitsune_cfg.get("max_autoencoder_size"),
                                       kd["max_autoencoder_size"], int, lo=2,
                                       name="engine.kitsune.max_autoencoder_size"),
        "fm_grace_period": _valid(kitsune_cfg.get("fm_grace_period"),
                                  kd["fm_grace_period"], int, lo=0,
                                  name="engine.kitsune.fm_grace_period"),
        "ad_grace_period": _valid(kitsune_cfg.get("ad_grace_period"),
                                  kd["ad_grace_period"], int, lo=0,
                                  name="engine.kitsune.ad_grace_period"),
        "learning_rate": _valid(kitsune_cfg.get("learning_rate"),
                                kd["learning_rate"], float, lo=1e-6,
                                name="engine.kitsune.learning_rate"),
        "threshold_percentile": _valid(kitsune_cfg.get("threshold_percentile"),
                                       kd["threshold_percentile"], float,
                                       lo=50.0001, hi=100.0,
                                       name="engine.kitsune.threshold_percentile"),
    }

    re_cfg = _as_mapping(engine, "rule_engine")
    rate_cfg = _as_mapping(re_cfg, "rate_limit")
    rd = _DEFAULT_ENGINE["rule_engine"]

    window_seconds = _valid(rate_cfg.get("window_seconds"), rd["window_seconds"],
                            float, lo=1e-6, name="engine.rule_engine.rate_limit.window_seconds")
    max_connections = _valid(rate_cfg.get("max_connections_per_window"),
                             rd["max_connections_per_window"], int, lo=1,
                             name="engine.rule_engine.rate_limit.max_connections_per_window")

    raw_protocols = re_cfg.get("allowed_protocols")
    if (isinstance(raw_protocols, list) and raw_protocols
            and all(isinstance(p, int) and not isinstance(p, bool) and 0 <= p <= 255
                    for p in raw_protocols)):
        allowed_protocols = list(raw_protocols)
    else:
        logger.warning("engine.rule_engine.allowed_protocols=%r is not a list of "
                       "protocol numbers; using default %r", raw_protocols,
                       rd["allowed_protocols"])
        allowed_protocols = list(rd["allowed_protocols"])

    raw_icmp_types = re_cfg.get("allowed_icmp_types", rd["allowed_icmp_types"])
    if isinstance(raw_icmp_types, list) and all(
            isinstance(t, int) and not isinstance(t, bool) and 0 <= t <= 255
            for t in raw_icmp_types):
        allowed_icmp_types = sorted(set(raw_icmp_types))
    else:
        logger.warning("engine.rule_engine.allowed_icmp_types=%r is not a list of "
                       "ICMP type numbers; using default %r", raw_icmp_types,
                       rd["allowed_icmp_types"])
        allowed_icmp_types = list(rd["allowed_icmp_types"])

    return {
        "kitsune": kitsune,
        "rule_engine": {
            "window_seconds": window_seconds,
            "max_connections": max_connections,
            "allowed_protocols": allowed_protocols,
            "allowed_icmp_types": allowed_icmp_types,
        },
    }


# Defaults mirror config/config.yaml's ``blocking:`` block.
_DEFAULT_BLOCKING = {
    "strikes_threshold": 5,
    "strikes_window": 300.0,
    "temp_ban_seconds": 600.0,
    "temp_ban_count_to_perm": 3,
    "table_max": 50_000,
}


def load_blocking_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``blocking`` block of config.yaml (escalation policy knobs).

    Same degradation policy as the other loaders, with per-key validation.
    """
    data = _load_mapping(path)
    blocking = _as_mapping(data, "blocking")
    d = _DEFAULT_BLOCKING
    return {
        "strikes_threshold": _valid(blocking.get("strikes_threshold"),
                                    d["strikes_threshold"], int, lo=1,
                                    name="blocking.strikes_threshold"),
        "strikes_window": _valid(blocking.get("strikes_window"),
                                 d["strikes_window"], float, lo=1e-6,
                                 name="blocking.strikes_window"),
        "temp_ban_seconds": _valid(blocking.get("temp_ban_seconds"),
                                   d["temp_ban_seconds"], float, lo=1e-6,
                                   name="blocking.temp_ban_seconds"),
        "temp_ban_count_to_perm": _valid(blocking.get("temp_ban_count_to_perm"),
                                         d["temp_ban_count_to_perm"], int, lo=1,
                                         name="blocking.temp_ban_count_to_perm"),
        "table_max": _valid(blocking.get("table_max"), d["table_max"], int, lo=1,
                            name="blocking.table_max"),
    }


# API surface defaults.  auth_token "" = authentication DISABLED (local
# development); the app logs a loud WARNING in that mode.  cors_origins
# default only trusts localhost origins.
_DEFAULT_API = {
    "auth_token": "",
    "cors_origins": ["http://localhost:8000", "http://127.0.0.1:8000"],
    "host": "0.0.0.0",
    "port": 8000,
}


def load_api_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``api`` block: auth token, CORS origins, host and port.

    ``NIPS_API_TOKEN`` overrides the YAML value so container deployments
    don't need the secret inside the image/volume.  ``cors_origins`` must be
    a list of strings and any ``"*"`` entry is dropped with a warning: a
    wildcard CORS on an API that can blacklist arbitrary IPs is a browser-
    origin bypass, and the README documents that ``"*"`` is unsupported.
    """
    data = _load_mapping(path)
    api = _as_mapping(data, "api")

    token = os.environ.get("NIPS_API_TOKEN")
    if token is None:
        token = api.get("auth_token", _DEFAULT_API["auth_token"])
    if not isinstance(token, str):
        logger.warning("api.auth_token=%r is not a string; authentication disabled", token)
        token = ""

    raw_origins = api.get("cors_origins")
    if raw_origins is None:
        cors = list(_DEFAULT_API["cors_origins"])
    elif isinstance(raw_origins, list) and all(isinstance(o, str) for o in raw_origins):
        cors = [o for o in raw_origins if o != "*"]
        if len(cors) != len(raw_origins):
            logger.warning('api.cors_origins contained "*"; wildcard origins are '
                           "rejected (the API can modify firewall rules)")
    else:
        logger.warning("api.cors_origins=%r is not a list of strings; using default %r",
                       raw_origins, _DEFAULT_API["cors_origins"])
        cors = list(_DEFAULT_API["cors_origins"])

    host = _valid(api.get("host"), _DEFAULT_API["host"], str, name="api.host")
    port = _valid(api.get("port"), _DEFAULT_API["port"], int, lo=1, hi=65535,
                  name="api.port")

    return {"auth_token": token, "cors_origins": cors, "host": host, "port": port}


# ``engine.ml`` mirrors config/config.yaml: the learning detectors as a group.
# Off by default — a deployment opts detection in, it is not sprung on it.
_DEFAULT_ML = {"enabled": False, "detectors": []}


def load_ml_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``engine.ml`` block — whether learning detectors decide, and
    which ones to mount.

    With ``enabled`` false nothing ML-related is imported or constructed: the
    rule engine judges alone and traffic it does not decide is allowed.  That is
    a deliberate posture, not a broken detector, so it must not trigger the
    fail-closed drop — which is what turning ML *on* opts into.

    ``detectors`` entries are ``{uses: kitsune | lucid | package.mod:Class,
    enabled: true, params: {}}``.  Built-ins read their own tuning from
    ``engine.kitsune`` / ``engine.lucid``; ``params`` goes to a third-party
    detector's ``configure()`` unchanged.
    """
    data = _load_mapping(path)
    ml = _as_mapping(_as_mapping(data, "engine"), "ml")

    enabled = ml.get("enabled", _DEFAULT_ML["enabled"])
    if not isinstance(enabled, bool):
        logger.warning("engine.ml.enabled=%r is not a bool; using %r",
                       enabled, _DEFAULT_ML["enabled"])
        enabled = _DEFAULT_ML["enabled"]

    raw = ml.get("detectors", _DEFAULT_ML["detectors"])
    if not isinstance(raw, list):
        logger.warning("engine.ml.detectors=%r is not a list; using %r",
                       raw, _DEFAULT_ML["detectors"])
        raw = _DEFAULT_ML["detectors"]

    detectors: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning("skipping engine.ml.detectors entry %r: not a mapping", item)
            continue
        uses = item.get("uses")
        if not isinstance(uses, str) or not uses:
            logger.warning("skipping engine.ml.detectors entry %r: no 'uses'", item)
            continue
        params = item.get("params") or {}
        if not isinstance(params, dict):
            logger.warning("detector %r: params=%r is not a mapping; using {}",
                           uses, params)
            params = {}
        flag = item.get("enabled", True)
        if not isinstance(flag, bool):
            logger.warning("detector %r: enabled=%r is not a bool; using true",
                           uses, flag)
            flag = True
        detectors.append({"uses": uses, "params": params, "enabled": bool(flag)})

    return {"enabled": enabled, "detectors": detectors}


# LUCID detector defaults mirror config/config.yaml's ``engine.lucid`` block.
_DEFAULT_LUCID = {
    "time_window": 10.0,
    "packets_per_flow": 10,
    "model_path": "",
}


def load_lucid_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``engine.lucid`` block (time window, flow length, model path).

    ``model_path`` empty = LUCID not deployed: callers must not register the
    detector.  Same degradation policy as the other loaders.
    """
    data = _load_mapping(path)
    lucid = _as_mapping(_as_mapping(data, "engine"), "lucid")
    d = _DEFAULT_LUCID

    model_path = lucid.get("model_path", d["model_path"])
    if not isinstance(model_path, str):
        logger.warning("engine.lucid.model_path=%r is not a string; LUCID disabled",
                       model_path)
        model_path = ""

    return {
        "time_window": _valid(lucid.get("time_window"), d["time_window"], float,
                              lo=1e-6, name="engine.lucid.time_window"),
        "packets_per_flow": _valid(lucid.get("packets_per_flow"), d["packets_per_flow"],
                                   int, lo=2, name="engine.lucid.packets_per_flow"),
        "model_path": model_path,
    }


# Repository root, used to resolve relative storage paths so a systemd unit
# with a different WorkingDirectory still finds the same database.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_STORAGE = {
    "events_db": "data/events.db",
    "max_rows": 200_000,
    "retention_days": 30.0,
    "queue_size": 10_000,
}


def load_storage_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``storage`` block (event database and retention knobs).

    Relative ``events_db`` paths resolve against the repository root.  An
    unusable database location does not disable persistence: the caller falls
    back to the in-memory ring and reports it through /api/v1/status.
    """
    data = _load_mapping(path)
    storage = _as_mapping(data, "storage")
    d = _DEFAULT_STORAGE

    db = storage.get("events_db", d["events_db"])
    if not isinstance(db, str) or not db.strip():
        logger.warning("storage.events_db=%r is not a usable path; using default %r",
                       db, d["events_db"])
        db = d["events_db"]
    db_path = Path(db.strip()).expanduser()
    if not db_path.is_absolute():
        db_path = _REPO_ROOT / db_path

    return {
        "events_db": str(db_path),
        "max_rows": _valid(storage.get("max_rows", d["max_rows"]), d["max_rows"], int, lo=1000,
                           name="storage.max_rows"),
        "retention_days": _valid(storage.get("retention_days", d["retention_days"]), d["retention_days"],
                                 float, lo=1e-6, name="storage.retention_days"),
        "queue_size": _valid(storage.get("queue_size", d["queue_size"]), d["queue_size"], int, lo=100,
                             name="storage.queue_size"),
    }


_DEFAULT_LOGGING = {
    "level": "INFO",
    "file": "",
    "max_bytes": 10_485_760,
    "backups": 5,
    "syslog_address": "",
}


def load_logging_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict:
    """Return the ``logging`` block (level, rotating file, syslog target).

    This block existed in config.yaml without any reader, so a configured
    level or syslog target silently did nothing.
    """
    data = _load_mapping(path)
    log = _as_mapping(data, "logging")
    d = _DEFAULT_LOGGING

    level = log.get("level", d["level"])
    if not isinstance(level, str) or not level.strip():
        logger.warning("logging.level=%r is not a string; using %r", level, d["level"])
        level = d["level"]

    file_target = log.get("file", d["file"])
    if file_target is None:
        file_target = ""
    if not isinstance(file_target, str):
        logger.warning("logging.file=%r is not a string; logging to console only",
                       file_target)
        file_target = ""
    if file_target.strip():
        resolved = Path(file_target.strip()).expanduser()
        if not resolved.is_absolute():
            resolved = _REPO_ROOT / resolved
        file_target = str(resolved)

    syslog_address = log.get("syslog_address", d["syslog_address"])
    if syslog_address is None:
        syslog_address = ""
    if not isinstance(syslog_address, str):
        logger.warning("logging.syslog_address=%r is not a string; syslog disabled",
                       syslog_address)
        syslog_address = ""

    return {
        "level": level.strip().upper(),
        "file": file_target.strip(),
        "max_bytes": _valid(log.get("max_bytes", d["max_bytes"]), d["max_bytes"], int, lo=1024,
                            name="logging.max_bytes"),
        "backups": _valid(log.get("backups", d["backups"]), d["backups"], int, lo=0, hi=100,
                          name="logging.backups"),
        "syslog_address": syslog_address.strip(),
    }
