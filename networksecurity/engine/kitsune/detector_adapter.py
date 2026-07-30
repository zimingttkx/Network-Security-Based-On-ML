"""Kitsune detector — adapted to the BaseDetector interface."""

from __future__ import annotations

import logging
from typing import Optional

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.kitsune.kitsune import Kitsune
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict

logger = logging.getLogger(__name__)


class KitsuneDetector(BaseDetector):
    """AfterImage + KitNET anomaly detector.

    Operates in two modes:
    - Training:  first ~55k packets build the normality model (no verdict).
    - Detection: after training, anomalous packets return BLOCK verdict.
    """

    def __init__(
        self,
        threshold_percentile: float = 99.0,
        max_autoencoder_size: int = 10,
    ) -> None:
        super().__init__(name="KitsuneDetector")
        self._kitsune = Kitsune(
            max_autoencoder_size=max_autoencoder_size,
            threshold_percentile=threshold_percentile,
        )

    # -- BaseDetector interface ---------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Optional[Verdict]:
        self._packet_count += 1

        result = self._kitsune.process_packet(packet.to_dict())

        if result.is_training:
            return None  # still learning

        if result.is_anomaly:
            confidence = min(1.0, result.rmse / max(0.001, (result.threshold or 1.0)))
            return Verdict(
                action=Action.BLOCK,
                confidence=confidence,
                threat_level=self._threat_level_from_rmse(result.rmse,
                                                          result.threshold or 1.0),
                reason=f"Kitsune anomaly (RMSE={result.rmse:.4f})",
                detector=self.name,
                metadata=result.to_dict(),
            )

        return None  # normal -> pass

    # -- helpers ------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._kitsune.is_ready()

    def get_state(self) -> dict:
        return self._kitsune.get_state()

    def reset(self) -> None:
        super().reset()
        self._kitsune.reset()

    @staticmethod
    def _threat_level_from_rmse(rmse: float, threshold: float) -> ThreatLevel:
        ratio = rmse / max(0.001, threshold)
        if ratio > 3.0:
            return ThreatLevel.CRITICAL
        if ratio > 2.0:
            return ThreatLevel.HIGH
        if ratio > 1.5:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW
