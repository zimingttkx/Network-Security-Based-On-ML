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
                value = float(params[key])
                if value < floor:
                    raise ValueError(f"{self.name}.{key}={value!r} is below {floor}")
                setattr(self, key, int(value) if default == int(default) else value)

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
