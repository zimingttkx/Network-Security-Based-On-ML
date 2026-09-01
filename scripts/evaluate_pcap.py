#!/usr/bin/env python3
"""End-to-end evaluation: real-data pcap -> full NIPS pipeline -> per-category report.

Uses the reconstructed UNSW-NB15 pcap (real dataset flows, real packet sizes
and timing).  Kitsune grace periods are shortened (documented set_grace_periods
path) so training completes within this capture; production defaults are 5k/50k.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.data.pcap_loader import PcapLoader
from networksecurity.engine import DetectionPipeline, PacketInfo
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
from networksecurity.interception.packet_parser import PacketParser

PCAP = "datasets/unsw-nb15/unsw_reconstructed.pcap"

# Ground truth: attack nets per the documented UNSW-NB15 testbed addressing
# used by scripts/build_unsw_pcap.py (175.45.176.0/22 attackers).
ATTACK_PREFIX = "175.45.176."


async def main():
    pipeline = DetectionPipeline()
    kitsune = KitsuneDetector()
    # Shortened grace (documented override) so 82k packets cover training+detection.
    kitsune.set_grace_periods(fm_grace_period=2000, ad_grace_period=30_000)
    pipeline.add_detector(kitsune)
    try:
        pipeline.add_detector(LucidDetectorAdapter(enabled=False))
    except ImportError:
        pass

    loader = PcapLoader()
    n = 0
    train_phase_end = 32_000  # fm+ad grace
    post = {"attack": 0, "normal": 0}
    post_block = {"attack": 0, "normal": 0}
    reasons: dict[str, int] = {}
    t0 = time.monotonic()

    async for pkt_dict in loader.load(PCAP):
        if pkt_dict is None:
            continue
        packet = PacketParser.from_dict(pkt_dict)
        verdict = await pipeline.process_packet(packet)
        n += 1
        if n <= train_phase_end:
            continue
        is_attack = packet.src_ip.startswith(ATTACK_PREFIX)
        cls = "attack" if is_attack else "normal"
        post[cls] += 1
        if verdict.action.value == "block":
            post_block[cls] += 1
            reasons[verdict.reason.split("(")[0].strip()] = reasons.get(
                verdict.reason.split("(")[0].strip(), 0) + 1
        if n % 20_000 == 0:
            print(f"  ... {n} packets, "
                  f"post-train blocked {sum(post_block.values())}/{sum(post.values())}",
                  flush=True)

    elapsed = time.monotonic() - t0
    tp, fn = post_block["attack"], post["attack"] - post_block["attack"]
    fp, tn = post_block["normal"], post["normal"] - post_block["normal"]
    tpr = tp / max(1, tp + fn) * 100
    fpr = fp / max(1, fp + tn) * 100
    precision = tp / max(1, tp + fp) * 100

    print("\n================ END-TO-END EVALUATION ================")
    print(f" pcap packets processed : {n} ({n/elapsed:.0f} pkt/s, {elapsed:.1f}s)")
    print(f" Kitsune trained        : {kitsune.is_ready}")
    print(f" evaluation window      : {sum(post.values())} packets (post-training)")
    print(f"   attack packets       : {post['attack']}")
    print(f"   normal packets       : {post['normal']}")
    print(f" TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f" detection rate (TPR)   : {tpr:.1f}%")
    print(f" false positive rate    : {fpr:.1f}%")
    print(f" precision              : {precision:.1f}%")
    print(f" block reasons          : {reasons}")
    print(f" pipeline counters      : processed={pipeline.total_processed} "
          f"blocked={pipeline.total_blocked}")
    print("=======================================================")


asyncio.run(main())
