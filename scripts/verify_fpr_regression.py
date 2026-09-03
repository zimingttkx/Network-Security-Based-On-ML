#!/usr/bin/env python3
"""FPR regression guard for the Kitsune detection path.

The wall-clock-feature incident (2026-09) shipped because CI only asserted
attack detection rate — a detector that flags 100% of traffic "detects" every
attack.  This script closes that hole: after training on normal traffic it
feeds fresh NORMAL traffic and asserts the false-positive rate stays low,
plus a sanity attack-detection check.

Exit code 0 = pass, 1 = fail (CI-visible).
"""
import asyncio
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from benchmark import TrafficGenerator as TG
from networksecurity.engine import DetectionPipeline, PacketInfo, Action
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector


async def main() -> int:
    random.seed(20260904)  # deterministic run

    pipeline = DetectionPipeline()
    kitsune = KitsuneDetector()
    # Short grace periods: the regression we guard against is threshold/
    # normalization drift, which manifests regardless of grace length.
    kitsune.set_grace_periods(fm_grace_period=500, ad_grace_period=1500)
    pipeline.add_detector(kitsune)

    # Phase 1: train on normal traffic (1ms spacing — matches detection rate
    # so the rate does not itself look anomalous).
    train_n = 2001
    for i in range(train_n):
        pkt = TG.normal_packet(timestamp=i * 0.001)
        await pipeline.process_packet(pkt)
    if not kitsune.is_ready:
        print("FAIL: Kitsune did not finish training")
        return 1

    # Phase 2: fresh normal traffic -> must mostly PASS
    det_n = 2000
    fp = 0
    base = train_n * 0.001
    for i in range(det_n):
        pkt = TG.normal_packet(timestamp=base + i * 0.001)
        v = await pipeline.process_packet(pkt)
        if v.action == Action.BLOCK:
            fp += 1
    fpr = fp / det_n * 100

    # Phase 3: SYN flood -> must mostly BLOCK (sanity check on the same model)
    atk_n = 500
    tp = 0
    atk_base = base + det_n * 0.001 + 5.0
    for i in range(atk_n):
        pkt = TG.syn_flood_packet(timestamp=atk_base + i * 0.001)
        v = await pipeline.process_packet(pkt)
        if v.action == Action.BLOCK:
            tp += 1
    tpr = tp / atk_n * 100

    print(f"FPR on normal traffic : {fp}/{det_n} ({fpr:.2f}%)  [assert < 5%]")
    print(f"TPR on SYN flood      : {tp}/{atk_n} ({tpr:.1f}%)  [assert >= 50%]")

    ok = fpr < 5.0 and tpr >= 50.0
    if not ok:
        print("FAIL: FPR/TPR regression detected")
        return 1
    print("PASS: no FPR/TPR regression")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
