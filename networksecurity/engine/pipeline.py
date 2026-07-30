"""Detection pipeline: chains detectors with short-circuit semantics."""

from __future__ import annotations

import logging
from typing import Optional

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action, Verdict

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Orchestrates multiple detectors in priority order.

    Short-circuit rule:
    - RuleEngine BLOCK   -> stop, return verdict
    - Any detector BLOCK  -> stop, return verdict (configurable)
    - None (pass)         -> continue to next detector
    - Final fallback      -> ALLOW
    """

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        short_circuit_on_block: bool = True,
    ) -> None:
        self._rule_engine = rule_engine or RuleEngine()
        self._detectors: list[BaseDetector] = [self._rule_engine]
        self._short_circuit_on_block = short_circuit_on_block
        self._running: bool = False
        self._total_processed: int = 0
        self._total_blocked: int = 0

    # -- registration -------------------------------------------------------

    def add_detector(self, detector: BaseDetector) -> DetectionPipeline:
        self._detectors.append(detector)
        return self

    @property
    def rule_engine(self) -> RuleEngine:
        return self._rule_engine

    @property
    def detectors(self) -> list[BaseDetector]:
        return self._detectors

    # -- processing ---------------------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict:
        self._total_processed += 1

        for detector in self._detectors:
            verdict = await detector.process_packet(packet)
            if verdict is None:
                continue
            if verdict.action == Action.BLOCK and self._short_circuit_on_block:
                self._total_blocked += 1
                return verdict
            if verdict.action != Action.BLOCK:
                return verdict

        return Verdict(
            action=Action.ALLOW,
            confidence=0.5,
            reason="no threat detected",
            detector="pipeline",
        )

    async def process_batch(self, packets: list[PacketInfo]) -> list[Verdict]:
        results = []
        for p in packets:
            results.append(await self.process_packet(p))
        return results

    # -- lifecycle ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def total_processed(self) -> int:
        return self._total_processed

    @property
    def total_blocked(self) -> int:
        return self._total_blocked

    def status(self) -> dict:
        return {
            "running": self._running,
            "total_processed": self._total_processed,
            "total_blocked": self._total_blocked,
            "detectors": [d.name for d in self._detectors],
            "rule_engine": self._rule_engine.stats(),
        }

    def reset(self) -> None:
        for d in self._detectors:
            d.reset()
        self._total_processed = 0
        self._total_blocked = 0
