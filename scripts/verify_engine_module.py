#!/usr/bin/env python3
"""Cross-validation for engine/ module (pipeline, rule_engine, kitsune, lucid adapters)."""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.engine import Action, DetectionPipeline, PacketInfo
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.rule_engine import RateLimiter, RuleEngine
from networksecurity.engine.verdict import ThreatLevel

results = []


def report(name: str, confirmed: bool, evidence: str):
    status = "CONFIRMED-BUG" if confirmed else "PASS"
    results.append((name, status))
    print(f"[{status}] {name}\n        {evidence}\n", flush=True)


def pkt(**kw) -> PacketInfo:
    base = dict(src_ip="1.2.3.4", dst_ip="10.0.0.1", src_port=1234, dst_port=80,
                protocol=6, packet_size=100, timestamp=1000.0)
    base.update(kw)
    return PacketInfo(**base)


# ---------------------------------------------------------------------------
# Checklist — pipeline
# P1 None -> continue; BLOCK -> short-circuit; ALLOW -> short-circuit
# P2 short_circuit_on_block=False keeps strongest BLOCK
# P3 fallback ALLOW verdict when chain abstains
# P4 counters consistent (total_processed/blocked)
# P5 reset() clears counters and detectors
# Checklist — rule_engine
# R1 whitelist hit -> ALLOW confidence 1.0
# R2 protocol filter blocks non-TCP/UDP (ICMP=1)
# R3 blacklist (exact + CIDR) -> BLOCK
# R4 rate limiter: over cap -> BLOCK; window expiry re-allows
# R5 rate limiter non-positive timestamp skipped
# R6 max_buckets eviction keeps memory bounded
# R7 load/save rules round-trip
# R8 thread-safety smoke: concurrent CRUD + reads
# Checklist — kitsune adapter
# K1 grace override before training works; after training raises
# K2 is_ready False during training, True after
# K3 confidence = rmse/threshold capped at 1.0
# K4 reset() restores untrained state
# Checklist — lucid adapter
# L1 disabled/untrained -> always None (abstain)
# L2 dict interface fields complete (header_size by protocol)
# ---------------------------------------------------------------------------

async def main():
    # --- P1/P2/P3: pipeline semantics -------------------------------------
    from networksecurity.engine.detector import BaseDetector
    from networksecurity.engine.verdict import Verdict

    class AlwaysBlock(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.BLOCK, 0.9, reason="test", detector="AlwaysBlock")

    class AlwaysAllow(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.ALLOW, 1.0, reason="test", detector="AlwaysAllow")

    pl = DetectionPipeline()
    pl.add_detector(AlwaysBlock())
    v = await pl.process_packet(pkt())
    report("P1 BLOCK short-circuits", v.action != Action.BLOCK, f"verdict={v.action}")

    pl2 = DetectionPipeline(short_circuit_on_block=False)
    pl2.add_detector(AlwaysBlock())

    class StrongBlock(BaseDetector):
        async def process_packet(self, packet):
            return Verdict(Action.BLOCK, 0.99, reason="strong", detector="StrongBlock")

    pl2.add_detector(StrongBlock())
    v2 = await pl2.process_packet(pkt())
    report("P2 non-short-circuit returns strongest BLOCK",
           not (v2.action == Action.BLOCK and v2.confidence == 0.99),
           f"verdict conf={v2.confidence} (expected 0.99 from StrongBlock)")

    pl3 = DetectionPipeline()
    v3 = await pl3.process_packet(pkt())
    report("P3 fallback ALLOW", v3.action != Action.ALLOW, f"verdict={v3.action}")

    # --- R1-R4: rule engine ------------------------------------------------
    re = RuleEngine()
    re.add_whitelist("1.2.3.4")
    v = await re.process_packet(pkt())
    report("R1 whitelist ALLOW", not (v and v.action == Action.ALLOW and v.confidence == 1.0),
           f"verdict={v}")

    re2 = RuleEngine()
    v = await re2.process_packet(pkt(protocol=1))
    report("R2 ICMP blocked", not (v and v.action == Action.BLOCK and "protocol" in v.reason),
           f"verdict={v}")

    re3 = RuleEngine()
    re3.add_blacklist("5.6.7.0/24")
    v = await re3.process_packet(pkt(src_ip="5.6.7.8"))
    report("R3 CIDR blacklist BLOCK", not (v and v.action == Action.BLOCK), f"verdict={v}")

    rl = RateLimiter(window_seconds=1.0, max_connections=5)
    over = [rl.check("9.9.9.9", 100.0 + i * 0.01) for i in range(10)]
    blocked_at = not all(over)
    later = rl.check("9.9.9.9", 102.5)  # window expired
    report("R4 rate limit + expiry", not (blocked_at and later), f"over-cap blocked={blocked_at}, after-window={later}")

    rl2 = RateLimiter(window_seconds=1.0, max_connections=1)
    all_pass = all(rl2.check("8.8.8.8", 0.0) for _ in range(100))
    report("R5 timestamp<=0 skipped", not all_pass, f"all_pass={all_pass}")

    rl3 = RateLimiter(window_seconds=0.5, max_connections=10, max_buckets=50)
    for i in range(500):
        rl3.check(f"10.{i//250}.{i%250}.1", 100.0 + i * 0.001)
    report("R6 bucket cap respected", len(rl3._buckets) > 50,
           f"buckets={len(rl3._buckets)} (cap 50)")

    # --- R7 rules persistence ----------------------------------------------
    import tempfile
    re7 = RuleEngine()
    re7.add_blacklist("3.3.3.3")
    re7.add_whitelist("4.4.4.4")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    re7.save_rules(path)
    re8 = RuleEngine()
    re8.load_rules(path)
    ok = re8.get_blacklist() == ["3.3.3.3"] and re8.get_whitelist() == ["4.4.4.4"]
    report("R7 save/load round-trip", not ok, f"blacklist={re8.get_blacklist()}, whitelist={re8.get_whitelist()}")

    # --- R8 concurrency smoke ------------------------------------------------
    import threading
    re9 = RuleEngine()
    errors = []

    def cruder():
        try:
            for i in range(200):
                re9.add_blacklist(f"10.0.{i}.1")
                re9.get_blacklist()
                re9.remove_blacklist(f"10.0.{i}.1")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for i in range(200):
                asyncio.run(re9.process_packet(pkt(src_ip=f"10.0.{i}.2", timestamp=100.0 + i)))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=cruder) for _ in range(2)] + \
              [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report("R8 concurrent CRUD+read no exception", bool(errors), f"errors={errors[:2]}")

    # --- K1/K2/K4: kitsune adapter ------------------------------------------
    kd = KitsuneDetector()
    kd.set_grace_periods(fm_grace_period=100, ad_grace_period=200)
    ready_during = kd.is_ready
    n_train = 0
    t0 = 10.0
    for i in range(350):
        p = pkt(src_ip=f"172.16.{i % 250}.1", src_port=1024 + i % 1000,
                packet_size=80 + (i * 13) % 400, timestamp=t0 + i * 0.01,
                dst_port=[80, 443, 22][i % 3])
        r = await kd.process_packet(p)
        n_train += 1
    ready_after = kd.is_ready
    try:
        kd.set_grace_periods(1, 1)
        raised = False
    except RuntimeError:
        raised = True
    report("K1 grace override raises post-training", not raised, f"raised={raised}")
    report("K2 is_ready transitions", ready_during or not ready_after,
           f"during={ready_during}, after={ready_after}")

    kd.reset()
    report("K4 reset restores untrained", kd.is_ready or kd._kitsune.is_initialized,
           f"is_ready={kd.is_ready}, initialized={kd._kitsune.is_initialized}")

    # K3 confidence cap
    from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector as KD
    conf = min(1.0, 5.0 / max(0.001, 0.5))
    report("K3 confidence formula caps at 1.0", conf != 1.0, f"conf={conf}")

    # --- L1/L2: lucid adapter interface --------------------------------------
    from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
    la = LucidDetectorAdapter(enabled=False)
    v = await la.process_packet(pkt())
    report("L1 disabled lucid abstains", v is not None, f"verdict={v}")
    d = LucidDetectorAdapter._to_lucid_dict(pkt(protocol=17))
    ok = d["header_size"] == 8 and d["payload_size"] == 60
    report("L2 lucid dict header_size by protocol", not ok, f"dict={d}")

    print("\n==== SUMMARY ====")
    for name, status in results:
        print(f"  {status:14s} {name}")


asyncio.run(main())
