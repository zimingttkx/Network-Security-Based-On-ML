#!/usr/bin/env python3
"""FPR regression guard for the Kitsune detection path.

The wall-clock-feature incident (2026-09) shipped because CI only asserted
attack detection rate — a detector that flags 100% of traffic "detects" every
attack.  This script closes that hole: after training on normal traffic it
feeds fresh NORMAL traffic and asserts the false-positive rate stays low,
reports the same traffic's cost with the rule engine alone, then measures that
rule engine against a single-source SYN flood.  The second half used to run
through the whole pipeline on addresses drawn from a 5 148-IP pool — one
packet per source, so the rate limiter never engaged and the "100% detected"
it printed was the anomaly detector flagging everything, the exact failure this
file exists to catch.

Exit code 0 = pass, 1 = fail (CI-visible).
"""
import asyncio
import random
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from benchmark import TrafficGenerator as TG
from networksecurity.engine import DetectionPipeline, Action, PacketInfo
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.kitsune.kitnet import KitNET


async def main() -> int:
    # Both RNGs matter: TrafficGenerator draws from `random`, while the
    # autoencoder weights are initialised with np.random.uniform.  Seeding
    # only the former left the anomaly threshold varying run to run (2.01 vs
    # 1.51 on identical traffic), so the gate flapped independently of the
    # code under test.
    random.seed(20260904)
    np.random.seed(20260904)

    pipeline = DetectionPipeline()
    kitsune = KitsuneDetector()
    # Short grace periods: the regression we guard against is threshold/
    # normalization drift, which manifests regardless of grace length.
    FM_GRACE, AD_GRACE = 500, 1500
    kitsune.set_grace_periods(fm_grace_period=FM_GRACE, ad_grace_period=AD_GRACE)
    pipeline.add_detector(kitsune)

    # Phase 1: train on normal traffic (1ms spacing — matches detection rate
    # so the rate does not itself look anomalous).  The threshold calibration
    # window is warm-up too, and it is derived from the AD grace, so the budget
    # has to be computed the same way rather than left at "grace plus one" —
    # otherwise this guard fails on `is_ready` for a reason unrelated to the
    # false-positive rate it exists to watch.
    calibrate_n = int(AD_GRACE * KitNET.CALIBRATION_FRACTION)
    train_n = FM_GRACE + AD_GRACE + calibrate_n + 1
    for i in range(train_n):
        pkt = TG.normal_packet(timestamp=i * 0.001)
        await pipeline.process_packet(pkt)
    if not kitsune.is_ready:
        print("FAIL: Kitsune did not finish training")
        return 1

    # Phase 2: fresh normal traffic -> must mostly PASS.  The same packets are
    # also fed to a rules-only pipeline so the README's "what does the rule
    # plane cost" row has a source in the repository.  That second number is
    # reported, not asserted: it is a reference measurement, and a gate that
    # cannot fail is documentation, not a gate.
    rules_only = DetectionPipeline(
        rule_engine=RuleEngine(window_seconds=1.0, max_connections=100),
        ml_enabled=False)
    det_n = 2000
    fp = 0
    rules_fp = 0
    base = train_n * 0.001
    for i in range(det_n):
        pkt = TG.normal_packet(timestamp=base + i * 0.001)
        v = await pipeline.process_packet(pkt)
        if v.action == Action.BLOCK:
            fp += 1
        rv = await rules_only.process_packet(pkt)
        if rv.action == Action.BLOCK:
            rules_fp += 1
    fpr = fp / det_n * 100
    rules_fpr = rules_fp / det_n * 100

    # Phase 3: a genuine single-source SYN flood -> the rate limiter must handle
    # it, with no ML in the path to take credit.  Two things had to be fixed to
    # measure that.  The old code drew attacker addresses from the 5 148-entry
    # ATTACK_IPS pool at 1 ms spacing — about one packet per source, so the
    # per-source cap was never approached and the "100% detected" it printed was
    # the anomaly detector flagging everything, which is the exact failure this
    # file exists to catch.  And it built the rule engine with bare defaults,
    # where ``max_connections`` is 1000 while `config/config.yaml` ships 100:
    # a guard measuring a configuration nobody deploys is not a guard.
    atk_n = 500
    tp = 0
    atk_base = base + det_n * 0.001 + 5.0
    for i in range(atk_n):
        pkt = PacketInfo(src_ip="203.0.113.66", dst_ip="10.0.0.1",
                        src_port=1024 + i, dst_port=80, protocol=6,
                        packet_size=40, tcp_flags=0x02,
                        timestamp=atk_base + i * 0.001)
        v = await rules_only.process_packet(pkt)
        if v.action == Action.BLOCK:
            tp += 1
    tpr = tp / atk_n * 100

    print(f"FPR, whole pipeline   : {fp}/{det_n} ({fpr:.2f}%)  [assert < 5%]")
    print(f"FPR, rule engine only : {rules_fp}/{det_n} ({rules_fpr:.2f}%)  "
          f"[reference, not asserted]")
    print(f"SYN flood, one source : {tp}/{atk_n} ({tpr:.1f}%)  "
          f"[assert >= 50%, rate limiter alone]")

    ok = fpr < 5.0 and tpr >= 50.0
    if not ok:
        print("FAIL: FPR/TPR regression detected")
        return 1
    print("PASS: no FPR/TPR regression")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
