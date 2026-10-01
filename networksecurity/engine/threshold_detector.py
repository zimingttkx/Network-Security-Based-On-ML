"""A complete worked example of the detector contract.

Mount it in config/config.yaml to replace the anomaly detectors with something
deterministic, or read it as the shape a third-party detector has to take:

    engine:
      ml:
        enabled: true
        detectors:
          - uses: networksecurity.engine.threshold_detector:ThresholdDetector
            params: {window_seconds: 5, max_packets: 1000}

Contract, in the order this class shows it: ``configure()`` takes tuning before
the first packet, ``process_packet()`` returns ``None`` to abstain or a Verdict
to decide, ``ready`` says whether this instance can score at all, and
``status()`` publishes whatever an operator would need to see.
"""

from __future__ import annotations

import math
from collections import OrderedDict

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict

_PARAMS = {"window_seconds": (5.0, 0.001), "max_packets": (1000, 1),
           "max_hosts": (50_000, 16)}


class ThresholdDetector(BaseDetector):
    """Block a source that exceeds N packets inside a rolling window."""

    def __init__(self, name: str = "ThresholdDetector"):
        super().__init__(name=name)
        self.window_seconds = _PARAMS["window_seconds"][0]
        self.max_packets = _PARAMS["max_packets"][0]
        self.max_hosts = _PARAMS["max_hosts"][0]
        self._stamps: OrderedDict[str, list[float]] = OrderedDict()
        self.trips = 0

    def configure(self, params: dict) -> None:
        unknown = sorted(set(params) - set(_PARAMS))
        if unknown:
            raise ValueError(f"{self.name} accepts {sorted(_PARAMS)}, got {unknown}")
        for key, (default, floor) in _PARAMS.items():
            if key in params:
                raw = params[key]
                # bool is an int subclass, so `max_packets: true` would otherwise
                # be stored as 1 — a config typo that silently disarms the
                # detector. utils/config.py refuses bools for the same reason.
                if isinstance(raw, bool):
                    raise ValueError(f"{self.name}.{key}={raw!r} is a bool, not a number")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    raise ValueError(f"{self.name}.{key}={raw!r} is not a number") from None
                # NaN compares False against every bound, so a finite check has
                # to come first: an infinite window would silently disable the
                # expiry below and keep a per-host history that never shrinks.
                if not math.isfinite(value) or value < floor:
                    raise ValueError(f"{self.name}.{key}={params[key]!r} is not a "
                                     f"finite value at or above {floor}")
                # Integer-valued knobs stay integers, fractional ones stay
                # fractional — classified by the *type* of the default.  The old
                # test was `default == int(default)`, which is also true for the
                # float default 5.0, so window_seconds went down the int branch:
                # 0.5 cleared its own floor and was then stored as 0, leaving a
                # detector that counts packets inside a zero-length window, never
                # trips, and still counts as ML coverage.
                stored = int(value) if isinstance(default, int) else value
                if stored < floor:
                    raise ValueError(f"{self.name}.{key}={params[key]!r} narrows to "
                                     f"{stored!r}, below the floor {floor}")
                setattr(self, key, stored)

    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        self._packet_count += 1
        stamps = self._stamps.get(packet.src_ip)
        if stamps is None:
            stamps = self._stamps[packet.src_ip] = []
        else:
            self._stamps.move_to_end(packet.src_ip)
        stamps.append(packet.timestamp)

        cutoff = packet.timestamp - self.window_seconds
        while stamps and stamps[0] < cutoff:
            stamps.pop(0)
        if not stamps:
            del self._stamps[packet.src_ip]

        if len(self._stamps) > self.max_hosts:
            self._stamps.popitem(last=False)

        if len(stamps) > self.max_packets:
            self.trips += 1
            return Verdict(
                action=Action.BLOCK,
                confidence=1.0,
                threat_level=ThreatLevel.MEDIUM,
                reason=f"{len(stamps)} packets in {self.window_seconds:g}s "
                       f"from {packet.src_ip}",
                detector=self.name,
                metadata={"count": len(stamps), "window_seconds": self.window_seconds},
            )
        return None

    def status(self) -> dict:
        return {"window_seconds": self.window_seconds,
                "max_packets": self.max_packets,
                "tracked_hosts": len(self._stamps),
                "trips": self.trips}
