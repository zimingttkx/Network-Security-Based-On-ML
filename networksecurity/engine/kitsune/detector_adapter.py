"""Kitsune detector — adapted to the BaseDetector interface."""

from __future__ import annotations

import logging

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
        learning_rate: float = 0.1,
    ) -> None:
        super().__init__(name="KitsuneDetector")
        self._kitsune = Kitsune(
            max_autoencoder_size=max_autoencoder_size,
            threshold_percentile=threshold_percentile,
            learning_rate=learning_rate,
        )

    @property
    def is_ready(self) -> bool:
        """Whether KitNET training has completed and detection is live."""
        return self._kitsune.is_ready

    @property
    def ready(self) -> bool:
        """True while training too: warm-up is not the same as being unable to
        score.  Excluding it would read as an outage and drop every packet
        during startup; ``status()`` is where the training state shows."""
        return True

    def status(self) -> dict:
        """`trained` is the honest "is it producing verdicts yet" flag."""
        return {"trained": bool(self._kitsune.is_ready)}

    # -- BaseDetector interface ---------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        self._packet_count += 1

        result = self._kitsune.process_packet(packet.to_dict())

        if result.is_training:
            return None  # still learning

        if result.is_anomaly:
            threshold = result.threshold or 1.0
            return Verdict(
                action=Action.BLOCK,
                confidence=self._confidence_from_rmse(result.rmse, threshold),
                threat_level=self._threat_level_from_rmse(result.rmse, threshold),
                reason=f"Kitsune anomaly (RMSE={result.rmse:.4f})",
                detector=self.name,
                metadata=result.to_dict(),
            )

        return None  # normal -> pass

    # -- helpers ------------------------------------------------------------

    def set_grace_periods(self, fm_grace_period: int | None = None,
                          ad_grace_period: int | None = None) -> None:
        """Override KitNET grace periods (before any packet is processed)."""
        self._kitsune.set_grace_periods(fm_grace_period, ad_grace_period)

    def get_state(self) -> dict:
        return self._kitsune.get_state()

    def reset(self) -> None:
        super().reset()
        self._kitsune.reset()

    @staticmethod
    def _confidence_from_rmse(rmse: float, threshold: float) -> float:
        """Map an anomaly score onto [0, 1].

        A BLOCK verdict only exists when ``rmse > threshold``, so ``rmse /
        threshold`` is always above 1 and clamps to a constant 1.0 — the
        previous formula reported maximum confidence for every anomaly,
        marginal and extreme alike.  Scaling by 0.5 puts the decision boundary
        at 0.5 and saturates at twice the threshold.
        """
        return min(1.0, 0.5 * rmse / max(0.001, threshold))

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
