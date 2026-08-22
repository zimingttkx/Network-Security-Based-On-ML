"""LUCID detector — adapted to the BaseDetector interface."""

from __future__ import annotations

import logging

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.lucid.detector import LucidDetector
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict

logger = logging.getLogger(__name__)


class LucidDetectorAdapter(BaseDetector):
    """LUCID CNN-based DDoS flow detector.

    Buffers packets into flows; only emits a verdict when a flow
    window (10 packets or 10s) completes.  Most packets return None.

    NOTE: LUCID is a *trained* model.  It must be trained (or have a
    pre-trained ``.h5`` model loaded) before it will emit BLOCK verdicts.
    Call ``self._lucid.train(...)`` or ``self._lucid.load(path)`` before
    deployment, otherwise ``process_packet`` always returns None.

    If no trained model is available at startup, pass ``enabled=False``
    to keep it out of the active detector chain and avoid advertising a
    detector that does nothing.
    """

    def __init__(
        self,
        time_window: float = 10.0,
        packets_per_flow: int = 10,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="LucidDetector")
        self._lucid = LucidDetector(
            time_window=time_window,
            packets_per_flow=packets_per_flow,
        )
        self._enabled = enabled

    # -- BaseDetector interface ---------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        if not self._enabled or not self._lucid.is_trained:
            return None  # not trained / disabled -> pass to next detector
        self._packet_count += 1

        # LucidDetector.process_packet expects a dict
        result = self._lucid.process_packet(self._to_lucid_dict(packet))
        if result is None:
            return None  # flow not complete yet

        if result.is_ddos:
            return Verdict(
                action=Action.BLOCK,
                confidence=result.confidence,
                threat_level=ThreatLevel.HIGH,
                reason=f"LUCID DDoS detected (conf={result.confidence:.2f})",
                detector=self.name,
                metadata=result.to_dict(),
            )

        return None

    # -- helpers ------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._enabled and self._lucid.is_trained

    def stats(self) -> dict:
        return self._lucid.get_stats()

    def reset(self) -> None:
        super().reset()
        self._lucid.reset_stats()

    @staticmethod
    def _to_lucid_dict(p: PacketInfo) -> dict:
        return {
            "src_ip": p.src_ip,
            "dst_ip": p.dst_ip,
            "src_port": p.src_port,
            "dst_port": p.dst_port,
            "protocol": p.protocol,  # integer: 6=TCP, 17=UDP
            "packet_size": p.packet_size,
            "timestamp": p.timestamp,
            "tcp_flags": p.tcp_flags,
            "direction": 0,
            "payload_size": p.payload_size or max(0, p.packet_size - 40),
            "header_size": 40 if p.protocol == 6 else (8 if p.protocol == 17 else 20),
            "window_size": 65535,
            "ttl": p.ttl,
        }
