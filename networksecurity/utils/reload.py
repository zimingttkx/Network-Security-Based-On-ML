"""Hot reload of the rule files and the engine knobs that can change live.

Two files are watched, for two different reasons:

* ``rules.json`` — operators edit it.  A delete must stop enforcing without a
  restart, which is why this uses ``reload_rules`` (replace) rather than
  ``load_rules`` (merge, correct at startup only).
* ``config/config.yaml`` — the rate-limit window/cap and the protocol
  allowlist are consulted per packet, so swapping them takes effect
  immediately.  Kitsune's grace periods and threshold percentile are *not*
  re-applied: they describe how the detector was trained, and changing them
  under a trained model would silently redefine what "normal" means.  The
  summary says so explicitly instead of pretending the whole file reloaded.

Every failure keeps the live state untouched and is counted, because a reload
that half-applies is worse than one that refuses.
"""

from __future__ import annotations

import logging
from pathlib import Path

from networksecurity.utils.config import load_engine_config, load_interception_config
from networksecurity.utils.validation import sweep_refused_entries

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


class ReloadProbe:
    """Compare file mtimes and apply what changed."""

    def __init__(self, rule_engine, rules_file: str | Path,
                 config_path: str | Path = _CONFIG_PATH) -> None:
        self.rule_engine = rule_engine
        self.rules_file = Path(rules_file)
        self.config_path = Path(config_path)
        self.reloads = 0
        self.failures = 0
        self.last_error = ""
        self.last_summary: dict | None = None
        # Seed with the current mtimes: the caller loaded these files at
        # startup, so the first probe must not report a change.
        self._rules_mtime = self._mtime(self.rules_file)
        self._config_mtime = self._mtime(self.config_path)

    @staticmethod
    def _mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def changed(self) -> dict[str, bool]:
        return {
            "rules": self._mtime(self.rules_file) != self._rules_mtime,
            "config": self._mtime(self.config_path) != self._config_mtime,
        }

    def probe(self, *, force: bool = False) -> dict | None:
        """Apply any change since the last probe.  None when nothing changed."""
        dirty = self.changed()
        if not force and not (dirty["rules"] or dirty["config"]):
            return None
        summary: dict = {"rules_file_changed": dirty["rules"],
                         "config_changed": dirty["config"],
                         "errors": []}
        if dirty["rules"] or force:
            self._apply_rules(summary)
        if dirty["config"] or force:
            self._apply_engine_knobs(summary)

        self._rules_mtime = self._mtime(self.rules_file)
        self._config_mtime = self._mtime(self.config_path)
        if summary["errors"]:
            self.failures += 1
            self.last_error = "; ".join(summary["errors"])
            logger.error("reload kept the previous state: %s", self.last_error)
        else:
            self.reloads += 1
            self.last_error = ""
        summary["engine_knobs_requiring_restart"] = [
            "engine.kitsune.fm_grace_period", "engine.kitsune.ad_grace_period",
            "engine.kitsune.threshold_percentile", "engine.kitsune.learning_rate"]
        self.last_summary = summary
        return summary

    def _apply_rules(self, summary: dict) -> None:
        try:
            counts = self.rule_engine.reload_rules(self.rules_file)
        except (OSError, ValueError) as exc:
            # ValueError covers json.JSONDecodeError and a bad entry; both mean
            # the live rule set stays exactly as it was.
            summary["errors"].append(f"rules.json: {exc}")
            summary["rules_applied"] = False
            return
        safe_ips = load_interception_config().get("safe_ips") or []
        swept = sweep_refused_entries(self.rule_engine, safe_ips)
        if swept:
            # Same policy as startup: an entry the kernel would refuse anyway is
            # dropped, otherwise it survives every restart as a phantom block.
            for ip in swept:
                logger.warning("reload dropped unenforceable blacklist entry %s", ip)
            self.rule_engine.save_rules(self.rules_file)
        summary["rules_applied"] = True
        summary["rules"] = counts
        summary["dropped_unenforceable"] = swept

    def _apply_engine_knobs(self, summary: dict) -> None:
        try:
            engine_cfg = load_engine_config(self.config_path)
        except Exception as exc:  # noqa: BLE001 - keep serving on the old knobs
            summary["errors"].append(f"config.yaml: {exc}")
            return
        knobs = engine_cfg["rule_engine"]
        self.rule_engine.set_rate_limit(knobs["window_seconds"], knobs["max_connections"])
        self.rule_engine.set_allowed_protocols(set(knobs["allowed_protocols"]))
        self.rule_engine.set_allowed_icmp_types(set(knobs["allowed_icmp_types"]))
        summary["rate_limit"] = {"window_seconds": knobs["window_seconds"],
                                 "max_connections": knobs["max_connections"]}
        summary["allowed_protocols"] = sorted(knobs["allowed_protocols"])
        summary["allowed_icmp_types"] = sorted(knobs["allowed_icmp_types"])

    def stats(self) -> dict:
        return {
            "reloads": self.reloads,
            "failures": self.failures,
            "last_error": self.last_error,
            "rules_file": str(self.rules_file),
            "watched": [str(self.rules_file), str(self.config_path)],
        }
