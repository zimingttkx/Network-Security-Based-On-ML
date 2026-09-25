#!/usr/bin/env python3
"""Cross-validation for the detector contract and the ML on/off switch.

Why this file exists: the chain used to decide fail-closed from "did any ML
detector get called", and an adapter that was registered but had no model still
answered with a LOG verdict — so it counted as coverage.  One live detector
tripping its breaker behind such an inert adapter switched the fail-closed
posture off silently and allowed everything.  These checks pin the distinction
between "switched off by config" (a decision: rules-only, allow) and "could not
run" (an outage: drop).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.engine import (Action, BaseDetector, DetectionPipeline,
                                    PacketInfo, RuleEngine)
from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
from networksecurity.engine.pipeline import DetectionUnavailable

results = []


def check(name: str, ok: bool, evidence: str):
    """report()-style helper: ok=False is the confirmed bug."""
    status = "PASS" if ok else "CONFIRMED-BUG"
    results.append((name, status))
    print(f"[{status}] {name}\n        {evidence}\n", flush=True)


def pkt(port: int = 80) -> PacketInfo:
    return PacketInfo(src_ip="1.2.3.4", dst_ip="10.0.0.1", src_port=1234,
                      dst_port=port, protocol=6, packet_size=100, timestamp=1000.0)


class Abstainer(BaseDetector):
    """A healthy ML detector that sees nothing wrong."""

    def __init__(self):
        super().__init__(name="Abstainer")
        self.calls = 0

    async def process_packet(self, packet):
        self.calls += 1
        return None


class Raiser(BaseDetector):
    """A capable detector that fails on every packet — the outage case."""

    def __init__(self):
        super().__init__(name="Raiser")
        self.calls = 0

    async def process_packet(self, packet):
        self.calls += 1
        raise RuntimeError("model file unreadable")


class ParamDetector(BaseDetector):
    def __init__(self):
        super().__init__(name="ParamDetector")
        self.got = None

    def configure(self, params: dict) -> None:
        self.got = dict(params)

    async def process_packet(self, packet):
        return None


def pipe_with(*detectors, **kw) -> DetectionPipeline:
    p = DetectionPipeline(RuleEngine(allowed_protocols={6, 17}), **kw)
    for d in detectors:
        p.add_detector(d)
    return p


async def main():
    # C1 — contract defaults
    d = Abstainer()
    refused = False
    try:
        d.configure({})
    except Exception:
        refused = True
    check("C1 contract defaults: ready True, empty status, configure({}) accepted",
          d.ready is True and d.status() == {} and not refused,
          f"ready={d.ready} status={d.status()!r} raised_on_empty={refused}")

    # C2 — a param for a detector that accepts none must be refused, not ignored
    plain = Abstainer()
    try:
        plain.configure({"threshold": 3})
        accepted, detail = True, "silently accepted the param"
    except ValueError as exc:
        accepted, detail = False, f"refused: {exc}"
    check("C2 unknown params rejected rather than ignored", not accepted, detail)

    # C3 — configure() delivers params to detectors that take them
    tuned = ParamDetector()
    tuned.configure({"window": 5})
    check("C3 configure() passes the config block through",
          tuned.got == {"window": 5}, f"got={tuned.got!r}")

    # C4 — ML switched off: the ML detector is never even called
    off_ml = Abstainer()
    v = await pipe_with(off_ml, ml_enabled=False).process_packet(pkt())
    check("C4 ml_enabled=False skips ML entirely and allows",
          off_ml.calls == 0 and v.action == Action.ALLOW,
          f"ml_calls={off_ml.calls} verdict={v.action.value}")

    # C5 — same chain with ML on: the detector is consulted
    on_ml = Abstainer()
    v = await pipe_with(on_ml, ml_enabled=True).process_packet(pkt())
    check("C5 ml_enabled=True consults ML and still allows on abstain",
          on_ml.calls == 1 and v.action == Action.ALLOW,
          f"ml_calls={on_ml.calls} verdict={v.action.value}")

    # C6 — rules-only because nothing can score: allow, do not drop
    inert = LucidDetectorAdapter(enabled=False)
    try:
        v = await pipe_with(inert).process_packet(pkt())
        dropped, verdict = False, v.action.value
    except DetectionUnavailable:
        dropped, verdict = True, None
    check("C6 disabled-by-config ML alone does not fail closed",
          (not dropped) and verdict == Action.ALLOW.value,
          f"dropped={dropped} verdict={verdict}")

    # C7 — the regression that mattered: a tripped live detector behind an inert
    # adapter must still fail closed.  Before the fix the inert adapter's LOG
    # verdict counted as coverage and every packet was allowed.
    live = Raiser()
    p = pipe_with(live, inert)
    dropped = 0
    for _ in range(8):
        try:
            await p.process_packet(pkt())
        except DetectionUnavailable:
            dropped += 1
    tripped = "Raiser" in p.status()["broken_detectors"]
    check("C7 tripped live detector fails closed even with an inert adapter present",
          tripped and dropped == 8, f"tripped={tripped} dropped={dropped}/8")

    # C8 — an inert adapter abstains; it no longer ends the chain with LOG
    v = await inert.process_packet(pkt())
    check("C8 inert adapter returns None instead of a chain-ending LOG",
          v is None, f"returned={v}")

    # C9 — status splits coverage honestly while ML is on
    live2 = Abstainer()
    s = pipe_with(live2, inert).status()
    check("C9 status reports coverage as only what can actually score",
          s["ml_enabled"] is True and s["ml_consulted"] == ["Abstainer"]
          and "LucidDetector" in s["ml_idle"]
          and s["degraded"] is False and s["ml_unavailable"] is False,
          f"consulted={s['ml_consulted']} idle={s['ml_idle']} "
          f"degraded={s['degraded']} unavailable={s['ml_unavailable']}")

    # C10 — with ML off, having nothing available is not an outage
    p = pipe_with(Raiser(), inert, ml_enabled=False)
    try:
        await p.process_packet(pkt())
        dropped = False
    except DetectionUnavailable:
        dropped = True
    s = p.status()
    check("C10 ml_enabled=False never drops and never claims an outage",
          (not dropped) and s["ml_enabled"] is False and s["degraded"] is False
          and s["ml_unavailable"] is False,
          f"dropped={dropped} degraded={s['degraded']} "
          f"unavailable={s['ml_unavailable']}")

    failed = [n for n, st in results if st == "CONFIRMED-BUG"]
    print("=" * 60)
    print(f"{len(results) - len(failed)}/{len(results)} PASS, "
          f"{len(failed)} CONFIRMED-BUG")
    for n in failed:
        print(f"  - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
