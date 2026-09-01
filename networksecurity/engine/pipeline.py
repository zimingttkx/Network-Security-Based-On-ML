"""Detection pipeline: chains detectors with short-circuit semantics."""

from __future__ import annotations

import logging
import threading

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action, Verdict

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Orchestrates multiple detectors in priority order.

    A detector returns either ``None`` (abstain, pass to the next detector)
    or an explicit ``Verdict``.  Any explicit verdict is a decision and
    short-circuits the chain:

    - ``None``            -> continue to the next detector
    - ``BLOCK``           -> stop and return it
    - ``ALLOW`` / ``LOG`` /
      ``CHALLENGE``       -> stop and return it (definitive decision)

    If ``short_circuit_on_block`` is ``False``, a ``BLOCK`` verdict does NOT
    stop the chain immediately; the pipeline keeps running the remaining
    detectors and returns the strongest observed ``BLOCK`` at the end (so
    later detectors can corroborate).  Non-BLOCK verdicts always stop the
    chain regardless of this flag, because an explicit allow/observe
    decision is final.

    If the chain finishes with no explicit verdict, the final fallback is
    ``ALLOW``.
    """

    def __init__(
        self,
        rule_engine: RuleEngine | None = None,
        short_circuit_on_block: bool = True,
    ) -> None:
        self._rule_engine = rule_engine or RuleEngine()
        self._detectors: list[BaseDetector] = [self._rule_engine]
        self._short_circuit_on_block = short_circuit_on_block
        self._running: bool = False
        self._lock = threading.Lock()
        self._total_processed: int = 0
        self._total_blocked: int = 0

    # -- registration -------------------------------------------------------

    def add_detector(self, detector: BaseDetector) -> DetectionPipeline:
        self._detectors.append(detector)
        return self

    def set_rule_engine(self, rule_engine: RuleEngine) -> DetectionPipeline:
        """Replace the default rule engine (position 0 in the chain).

        The pipeline constructs a default ``RuleEngine``; callers that read
        engine tuning from config.yaml use this to swap in a configured one
        before any packet is processed.
        """
        self._rule_engine = rule_engine
        self._detectors[0] = rule_engine
        return self

    @property
    def rule_engine(self) -> RuleEngine:
        return self._rule_engine

    @property
    def detectors(self) -> list[BaseDetector]:
        return self._detectors

    # -- processing ---------------------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict:
        with self._lock:
            self._total_processed += 1

        # When short-circuit is disabled we still want to capture a BLOCK
        # verdict even if a later detector overrides it; keep the strongest
        # (highest-confidence) BLOCK seen so far.
        pending_block: Verdict | None = None

        for detector in self._detectors:
            verdict = await detector.process_packet(packet)
            if verdict is None:
                continue

            if verdict.action == Action.BLOCK:
                with self._lock:
                    self._total_blocked += 1
                if self._short_circuit_on_block:
                    return verdict
                if pending_block is None or verdict.confidence > pending_block.confidence:
                    pending_block = verdict
                continue

            # Any non-BLOCK explicit verdict is a final decision and stops
            # the chain (ALLOW / LOG / CHALLENGE are definitive).
            return verdict

        if pending_block is not None:
            return pending_block

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

    def start(self) -> None:
        """Mark the pipeline as running (called by Interceptor on start)."""
        self._running = True

    def stop(self) -> None:
        """Mark the pipeline as stopped (called by Interceptor on stop)."""
        self._running = False

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
        with self._lock:
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
        with self._lock:
            self._total_processed = 0
            self._total_blocked = 0
