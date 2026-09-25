"""Detection pipeline: chains detectors with short-circuit semantics."""

from __future__ import annotations

import logging
import threading

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action, Verdict

logger = logging.getLogger(__name__)


class DetectionUnavailable(RuntimeError):
    """No ML detector was able to run on a packet.

    Raised instead of returning the ``ALLOW`` fallback when ML detectors are
    registered but none of them executed — every one is either tripped by the
    circuit breaker or raised on this packet.  The inline IPS is fail-closed:
    the interceptor catches this and drops the in-flight packet.  Returning
    ALLOW instead used to turn a total detector outage into a silent
    fail-open, waving every packet through at confidence 0.5 while the only
    visible symptom was a ``broken_detectors`` list nobody watched.

    Not raised when the chain reaches a deterministic decision (a rule-engine
    BLOCK/ALLOW, or any explicit ML verdict) — those are real verdicts, not
    the absence of one.
    """


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

    If the chain finishes with no explicit verdict the fallback is ``ALLOW``
    — but only when at least one ML detector actually ran.  When ML
    detectors are registered and none of them executed, ``process_packet``
    raises ``DetectionUnavailable`` so the caller fail-closes (see above).

    Fault isolation: a detector that raises is treated as abstaining for
    that packet, and after ``FAILURE_THRESHOLD`` consecutive exceptions it is
    skipped entirely (circuit breaker) so one poison detector cannot take
    down the whole chain — see process_packet().
    """

    # Consecutive exceptions before a detector is skipped for the rest of
    # the process lifetime.  Without this, one raising detector (corrupt
    # model file, OOM) made every process_packet() raise, and the
    # interceptor's catch-all then dropped EVERY packet — a self-DoS that
    # looked like "the network is down".
    FAILURE_THRESHOLD = 5

    def __init__(
        self,
        rule_engine: RuleEngine | None = None,
        short_circuit_on_block: bool = True,
        ml_enabled: bool = True,
    ) -> None:
        self._rule_engine = rule_engine or RuleEngine()
        self._detectors: list[BaseDetector] = [self._rule_engine]
        self._short_circuit_on_block = short_circuit_on_block
        # Runs the rule engine alone when False.  Distinct from "ML is broken":
        # switching detection off is a decision, so undecided traffic is allowed;
        # having it registered and unavailable is an outage, so undecided traffic
        # is dropped.
        self._ml_enabled = ml_enabled
        self._running: bool = False
        self._lock = threading.Lock()
        self._total_processed: int = 0
        self._total_blocked: int = 0
        # Per-detector consecutive-failure counts and the tripped set.  Only
        # the detection event-loop thread mutates these; status() snapshots.
        self._detector_failures: dict[str, int] = {}
        self._broken_detectors: set[str] = set()

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

    @property
    def ml_enabled(self) -> bool:
        return self._ml_enabled

    def set_ml_enabled(self, enabled: bool) -> None:
        """Run the rule engine alone, or bring the learning detectors back.

        Applies to the next packet — nothing is buffered, so this is a decision
        about future traffic rather than a revision of the past.
        """
        self._ml_enabled = bool(enabled)

    def _ml_detectors(self) -> list[BaseDetector]:
        """Registered detectors other than the rule engine.

        The rule engine sits at index 0 and is deterministic; everything
        behind it is a learning detector whose outage has to fail closed
        rather than fall through to ALLOW.
        """
        return [d for d in self._detectors if d is not self._rule_engine]

    # -- processing ---------------------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict:
        with self._lock:
            self._total_processed += 1

        # When short-circuit is disabled we still want to capture a BLOCK
        # verdict even if a later detector overrides it; keep the strongest
        # (highest-confidence) BLOCK seen so far.
        pending_block: Verdict | None = None
        # Did any ML detector actually run?  An abstain (None) or a LOG
        # verdict still counts as "ran" — the detector was healthy and made a
        # (negative) decision.  Only breaker-skips and exceptions leave it
        # False, and that distinction is what separates "nothing detected"
        # from "nothing could look".
        ml_executed = False

        for detector in self._detectors:
            if detector.name in self._broken_detectors:
                continue
            if not self._ml_enabled and detector is not self._rule_engine:
                continue
            try:
                verdict = await detector.process_packet(packet)
            except Exception:
                fails = self._detector_failures.get(detector.name, 0) + 1
                self._detector_failures[detector.name] = fails
                if fails >= self.FAILURE_THRESHOLD:
                    self._broken_detectors.add(detector.name)
                    logger.error(
                        "detector %s tripped the circuit breaker after %d "
                        "consecutive failures — skipping it until restart; "
                        "once every ML detector is skipped the pipeline "
                        "fail-closes (DetectionUnavailable -> packet dropped)",
                        detector.name, fails,
                    )
                else:
                    logger.exception(
                        "detector %s failed (%d/%d consecutive) — abstaining",
                        detector.name, fails, self.FAILURE_THRESHOLD,
                    )
                continue
            self._detector_failures.pop(detector.name, None)
            if detector is not self._rule_engine:
                ml_executed = True
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

        if self._ml_enabled and not ml_executed and self._ml_detectors():
            raise DetectionUnavailable(
                "no ML detector could run on this packet "
                f"(broken: {sorted(self._broken_detectors) or 'all raised'})"
            )

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
            broken = sorted(self._broken_detectors)
            ml = self._ml_detectors()
            down = [d.name for d in ml if d.name in self._broken_detectors]
            # "consulted" is what an operator actually has: enabled, not
            # tripped, and able to score.  A registered-but-inert adapter (no
            # model loaded) or one still training is not coverage, and listing
            # it beside the ones that are deciding traffic reads as a promise
            # the process is not keeping.
            consulted = [d.name for d in ml
                         if self._ml_enabled and d.name not in down
                         and getattr(d, "ready", True)]
            idle = [d.name for d in ml if d.name not in consulted]
            return {
                "running": self._running,
                "total_processed": self._total_processed,
                "total_blocked": self._total_blocked,
                "detectors": [d.name for d in self._detectors],
                "ml_enabled": self._ml_enabled,
                "ml_consulted": consulted,
                "ml_idle": idle,
                "broken_detectors": broken,
                # degraded: some ML coverage lost.  ml_unavailable: every
                # registered ML detector is tripped, so any packet that the
                # rule engine does not decide raises DetectionUnavailable and
                # is dropped — a full outage, not a partial one.  Both describe
                # unplanned loss, so neither fires while ML is switched off:
                # that is a decision, and the status says so with ml_enabled.
                "degraded": self._ml_enabled and bool(down),
                "ml_unavailable": bool(self._ml_enabled and ml
                                       and len(down) == len(ml)),
                "rule_engine": self._rule_engine.stats(),
            }

    def reset(self) -> None:
        for d in self._detectors:
            d.reset()
        self._detector_failures.clear()
        self._broken_detectors.clear()
        with self._lock:
            self._total_processed = 0
            self._total_blocked = 0
