#!/usr/bin/env python3
"""End-to-end evaluation: real-data pcap -> full NIPS pipeline -> per-category report.

Ground truth is an input, never a guess.  Labels come from the capture's
sidecar (``--labels``, defaulting to the same-named ``.labels.json``) — the
answer key the builder wrote next to the capture.  The evaluator must not infer
a label from packet contents: recovering truth from the source address an
attack came from is exactly what made the first version of this measurement
circular, and it is the one failure mode this file is written to make
impossible.  With ``--no-labels`` it reports verdicts and block reasons only,
which is what can honestly be measured on a capture you have no truth for.

A sidecar that describes none of the capture is refused for the same reason: an
empty labelled set scores 0.0% false positives and 0.0% detection, which is what
a perfect detector scores, so printing rates over it would be a report no one
can tell apart from success.

Both directions of a flow inherit that flow's label: a labelled attack's server
responses are attack packets too.  They used to be scored as normal traffic
(their source address was the victim's), which quietly padded the normal class
and counted every block on a response as a false positive.

Training runs at the shipped grace periods (``config/config.yaml``: 5k feature
mapping + 50k detection).  The shorter window this script used to configure — to
fit training and a detection window inside one 82k-packet capture — also fits the
feature map on 2 000 packets, which is whatever happened to arrive first; on this
capture that is a few long flows, and the operating point then moves with the
draw rather than with the detector (two draws of the same 1 680 flows: 1.3% and
52% of packets blocked).  Both draws agree at the shipped values.

The rates are also not reproducible run to run: KitNET initialises its weights
from the global RNG, which alone moves the false-positive rate on this capture
from 2.1% to 10.5% across twelve seeds, so ``--seed`` pins a draw when a number
needs to be quoted and ``verify_real_capture_quality.py`` scores three fixed
draws instead of relying on one.
"""
import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture_labels import flow_key, read_sidecar, sidecar_path
from networksecurity.data.pcap_loader import PcapLoader
from networksecurity.engine import DetectionPipeline
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
from networksecurity.interception.packet_parser import PacketParser

DEFAULT_PCAP = "datasets/unsw-nb15/unsw_reconstructed.pcap"
FM_GRACE, AD_GRACE = 5_000, 50_000  # config/config.yaml defaults


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pcap", default=DEFAULT_PCAP, help="capture to evaluate")
    ap.add_argument("--labels", default=None,
                    help="label sidecar (default: the capture's .labels.json)")
    ap.add_argument("--no-labels", action="store_true",
                    help="no ground truth available: report verdicts, never rates")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N packets (0 = the whole capture)")
    ap.add_argument("--fm-grace", type=int, default=FM_GRACE,
                    help="feature-map grace in packets (shortening it makes the "
                         "fitted feature map depend on which flows arrive first)")
    ap.add_argument("--ad-grace", type=int, default=AD_GRACE,
                    help="detection-training grace in packets")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed numpy before building the detector; KitNET's "
                         "autoencoders are otherwise initialised from the global "
                         "RNG, so the same capture scores differently each run")
    return ap.parse_args()


async def _evaluate(args: argparse.Namespace) -> int:
    if args.no_labels and args.labels:
        print("ERROR: --no-labels and --labels are mutually exclusive", file=sys.stderr)
        return 2

    labels_by_key = None
    labels_path = None
    if not args.no_labels:
        labels_path = Path(args.labels) if args.labels else sidecar_path(args.pcap)
        if not labels_path.exists():
            print(f"ERROR: no label sidecar at {labels_path}.  Ground truth is an "
                  f"input: pass --labels FILE, or --no-labels to report verdicts "
                  f"without rates.", file=sys.stderr)
            return 2
        _, labels_by_key = read_sidecar(labels_path)

    train_end = args.fm_grace + args.ad_grace
    if args.seed is not None:
        np.random.seed(args.seed)
    pipeline = DetectionPipeline()
    kitsune = KitsuneDetector()
    # Grace periods must be set before the first packet is processed.
    kitsune.set_grace_periods(fm_grace_period=args.fm_grace,
                              ad_grace_period=args.ad_grace)
    pipeline.add_detector(kitsune)
    try:
        pipeline.add_detector(LucidDetectorAdapter(enabled=False))
    except ImportError:
        pass

    loader = PcapLoader()
    n = 0
    counted = 0
    unmatched = 0
    blocked_total = 0
    post = {"attack": 0, "normal": 0}
    post_block = {"attack": 0, "normal": 0}
    per_cat: dict[str, list[int]] = {}  # category -> [blocked, total]
    reasons: Counter[str] = Counter()
    t0 = time.monotonic()

    async for pkt_dict in loader.load(args.pcap):
        if pkt_dict is None:
            continue
        packet = PacketParser.from_dict(pkt_dict)
        verdict = await pipeline.process_packet(packet)
        n += 1

        record = None
        if labels_by_key is not None:
            record = labels_by_key.get(flow_key(packet.protocol, packet.src_ip,
                                                packet.src_port, packet.dst_ip,
                                                packet.dst_port))
            if record is None:
                unmatched += 1

        if n > train_end:
            counted += 1
            blocked = verdict.action.value == "block"
            if blocked:
                blocked_total += 1
                reasons[verdict.reason.split("(")[0].strip()] += 1
            if record is not None:
                cls = "attack" if record["label"] else "normal"
                post[cls] += 1
                if blocked:
                    post_block[cls] += 1
                if cls == "attack":
                    stats = per_cat.setdefault(record["attack_cat"] or "attack", [0, 0])
                    stats[1] += 1
                    if blocked:
                        stats[0] += 1
        if n % 20_000 == 0:
            print(f"  ... {n} packets, post-train blocked {blocked_total}/{counted}",
                  flush=True)
        if args.limit and n >= args.limit:
            break

    elapsed = time.monotonic() - t0
    print("\n================ END-TO-END EVALUATION ================")
    print(f" pcap packets processed : {n} ({n/elapsed:.0f} pkt/s, {elapsed:.1f}s)")
    print(f" Kitsune trained        : {kitsune.is_ready}")

    if labels_by_key is None:
        print(" ground truth           : none (--no-labels) — verdicts only, no rates")
        print(f" post-training packets  : {counted}")
        print(f" post-training blocks   : {blocked_total}")
        print(f" block reasons          : {dict(reasons)}")
    else:
        if post["attack"] + post["normal"] == 0:
            # An empty labelled set scores 0.0% false positives and 0.0%
            # detection — the same numbers a perfect detector produces.  A
            # sidecar that drifted off this capture (or a --limit inside the
            # grace periods) must end here rather than print those.
            print(f"ERROR: no packet in the scoring window carries a label "
                  f"({counted} scored, {unmatched} unmatched against {labels_path}) — "
                  f"rates over an empty labelled set would be fiction.  Check "
                  f"--labels, or raise --limit above the grace periods "
                  f"({args.fm_grace} + {args.ad_grace}).", file=sys.stderr)
            return 2
        tp, fn = post_block["attack"], post["attack"] - post_block["attack"]
        fp, tn = post_block["normal"], post["normal"] - post_block["normal"]
        tpr = tp / max(1, tp + fn) * 100
        fpr = fp / max(1, fp + tn) * 100
        precision = tp / max(1, tp + fp) * 100
        print(f" labels                 : {labels_path} "
              f"({len(labels_by_key)} flows)")
        print(f" evaluation window      : {sum(post.values())} packets (post-training)")
        print(f"   attack packets       : {post['attack']}")
        print(f"   normal packets       : {post['normal']}")
        print(f" unmatched packets      : {unmatched}")
        print(f" TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        print(f" detection rate (TPR)   : {tpr:.1f}%")
        print(f" false positive rate    : {fpr:.1f}%")
        print(f" precision              : {precision:.1f}%")
        if per_cat:
            print(" per category           :")
            for cat, (blocked, total) in sorted(per_cat.items()):
                print(f"   {cat:<14} {blocked}/{total} attack packets blocked "
                      f"({blocked / max(1, total) * 100:.1f}%)")
        print(f" block reasons          : {dict(reasons)}")
    print(f" pipeline counters      : processed={pipeline.total_processed} "
          f"blocked={pipeline.total_blocked}")
    print("=======================================================")
    return 0


def main() -> int:
    return asyncio.run(_evaluate(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
