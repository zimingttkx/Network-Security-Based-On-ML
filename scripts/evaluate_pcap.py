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
can tell apart from success.  A sidecar covering only one class is refused too:
the rate whose denominator is empty is fiction either way, and a window of five
attack packets with no normal ones used to print "false positive rate 0.0%".

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
from networksecurity.engine.kitsune.kitnet import KitNET
from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
from networksecurity.interception.packet_parser import PacketParser

DEFAULT_PCAP = "datasets/unsw-nb15/unsw_reconstructed.pcap"
FM_GRACE, AD_GRACE = 5_000, 50_000  # config/config.yaml defaults


def _print_drift_buckets(samples: list[tuple[float, bool]], buckets: int,
                         scored: int) -> bool:
    """Report the flagged fraction per equal-width time bucket.

    Aggregate flag rate cannot tell the two failure shapes apart: a threshold
    calibrated on the wrong distribution is wrong from the first post-training
    packet, while drift after the freeze starts low and climbs.  Printed after
    every field the nightly gate parses, and deliberately carrying none of
    those labels, so adding it changes no existing reading.

    ``scored`` is the caller's authoritative count of post-training packets it
    judged.  The shape of this series is the evidence for a calibration
    decision, so it has to reconcile against that number rather than against
    the list it was handed: a list that quietly lost packets — or kept only the
    blocked ones — would otherwise print a perfectly self-consistent profile
    describing a different run.  False means the profile is not to be trusted.
    """
    if not samples:
        if scored:
            print(f"   BUCKETING BUG: buckets hold 0 of {scored} scored packets")
            return False
        return True
    # min/max, not first/last: pcap records are not guaranteed monotonic, and a
    # record outside [samples[0], samples[-1]] would land in no bucket at all —
    # a drift profile that quietly adds up to less than it scored.
    times = [t for t, _ in samples]
    first, last = min(times), max(times)
    span = last - first
    if span <= 0:  # a capture with one timestamp: bucket by arrival order
        edges = [len(samples) * i // buckets for i in range(buckets + 1)]
        groups = [(samples[edges[i]:edges[i + 1]], i * 1.0) for i in range(buckets)]
    else:
        groups = []
        for i in range(buckets):
            lo, hi = first + span * i / buckets, first + span * (i + 1) / buckets
            rows = [s for s in samples if lo <= s[0] < hi or (i == buckets - 1 and s[0] == last)]
            groups.append((rows, lo - first))
    counted = sum(len(rows) for rows, _ in groups)
    ok = counted == scored and len(samples) == scored
    print(f" drift profile          : {buckets} buckets over {span:.1f}s "
          f"of post-training traffic "
          f"({counted}/{scored} scored packets accounted for)")
    if not ok:
        print(f"   BUCKETING BUG: buckets hold {counted} of {scored} scored "
              f"packets ({len(samples)} collected into the series)")
    shown = 0
    for (rows, offset), i in zip(groups, range(buckets)):
        if not rows:
            continue
        flagged = sum(1 for _, f in rows if f)
        print(f"   bucket {i + 1:>2} t+{offset:>7.1f}s  {flagged:>6}/{len(rows):<6} "
              f"flagged ({flagged / len(rows) * 100:.1f}%)")
        shown += 1
    if shown < 2:
        print("   (fewer than two populated buckets — no shape to read)")
    return ok


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
    ap.add_argument("--threshold-percentile", type=float, default=99.0,
                    help="anomaly threshold percentile over the training RMSEs "
                         "(lower = more sensitive: higher detection rate and "
                         "higher false-positive rate)")
    ap.add_argument("--calibration-packets", type=int, default=None,
                    help="packets scored after the normalization freeze, whose "
                         "percentile becomes the anomaly threshold (default: 10% "
                         "of --ad-grace).  0 restores the legacy threshold taken "
                         "from the training pass, which is what the calibration "
                         "is compared against")
    ap.add_argument("--buckets", type=int, default=0,
                    help="split the post-training window into N time buckets and "
                         "report the flagged fraction per bucket (0 = off; the "
                         "default output is then byte-identical).  A flat profile "
                         "says the threshold was mis-calibrated from the first "
                         "packet; a rising one says the model drifted after it "
                         "froze")
    return ap.parse_args()


async def _evaluate(args: argparse.Namespace) -> int:
    if args.no_labels and args.labels:
        print("ERROR: --no-labels and --labels are mutually exclusive", file=sys.stderr)
        return 2
    if args.buckets < 0:
        print("ERROR: --buckets must be 0 (off) or a positive count", file=sys.stderr)
        return 2
    # Out of range here, not as an np.percentile ValueError after the capture has
    # been trained on: NaN fails the same comparison, so it is refused too.
    if not 0.0 <= args.threshold_percentile <= 100.0:
        print("ERROR: --threshold-percentile must be within [0, 100]", file=sys.stderr)
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
    kitsune = KitsuneDetector(threshold_percentile=args.threshold_percentile,
                              calibration_packets=args.calibration_packets)
    # Grace periods must be set before the first packet is processed.
    kitsune.set_grace_periods(fm_grace_period=args.fm_grace,
                              ad_grace_period=args.ad_grace)
    # The calibration window is a third warm-up, not a scoring window: the
    # detector abstains through it by design, so charging its packets as misses
    # would grade the detector on traffic it was never asked to judge.
    cal = (args.calibration_packets if args.calibration_packets is not None
           else int(args.ad_grace * KitNET.CALIBRATION_FRACTION))
    cal = max(0, cal)
    train_end += cal
    pipeline.add_detector(kitsune)
    try:
        pipeline.add_detector(LucidDetectorAdapter(enabled=False))
    except ImportError:
        pass

    loader = PcapLoader()
    n = 0
    counted = 0
    unmatched = 0
    unmatched_window = 0
    blocked_total = 0
    post = {"attack": 0, "normal": 0}
    post_block = {"attack": 0, "normal": 0}
    per_cat: dict[str, list[int]] = {}  # category -> [blocked, total]
    reasons: Counter[str] = Counter()
    # (timestamp, flagged) per post-training packet, kept only for --buckets:
    # the drift-vs-miscalibration question is answered by the *shape* of this
    # series, which the aggregate flag rate above cannot show.
    samples: list[tuple[float, bool]] = []
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
            if labels_by_key is not None and record is None:
                # The partial-truth refusal below quotes this window, so its
                # numbers have to add up inside it.  Run-wide `unmatched` also
                # counts warm-up packets, which are never scored and never
                # labelled, so mixing the two made a refusal print 5 + 178 for a
                # window of 179.
                unmatched_window += 1
            blocked = verdict.action.value == "block"
            if args.buckets:
                samples.append((packet.timestamp, blocked))
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
    print(f" calibration window     : {cal} packets "
          f"(threshold from {kitsune.status()['threshold_source']})")

    if labels_by_key is None:
        print(" ground truth           : none (--no-labels) — verdicts only, no rates")
        print(f" post-training packets  : {counted}")
        print(f" post-training blocks   : {blocked_total}")
        print(f" block reasons          : {dict(reasons)}")
    else:
        missing = [cls for cls, total in (("attack", post["attack"]),
                                          ("normal", post["normal"])) if total == 0]
        if missing:
            # An empty labelled set scores 0.0% false positives and 0.0%
            # detection — the numbers a perfect detector produces.  A *partial*
            # one is the same trap in disguise: a sidecar matching five attack
            # packets and no normal ones prints "false positive rate 0.0%" over
            # an empty class, and nothing on the page tells a reader that half
            # the ratio had no denominator.  Truth is an input, so a class nobody
            # labelled ends the run instead of feeding one of the two rates.
            # (`verify_real_capture_quality.py` already treats both halves of
            # this as a failure for the bundled capture; this moves the
            # invariant to the program that prints the number.)
            print(f"ERROR: the scoring window carries no {' and no '.join(missing)} "
                  f"labels ({counted} packets scored, {post['attack']} attack + "
                  f"{post['normal']} normal labelled, {unmatched_window} unmatched in "
                  f"the window, {unmatched} across the run, against {labels_path}) — "
                  f"rates over a partially labelled window would be "
                  f"fiction.  Check --labels, or raise --limit above the warm-up "
                  f"({args.fm_grace} + {args.ad_grace} + {cal} calibration).",
                  file=sys.stderr)
            return 2
        tp, fn = post_block["attack"], post["attack"] - post_block["attack"]
        fp, tn = post_block["normal"], post["normal"] - post_block["normal"]
        tpr = tp / max(1, tp + fn) * 100
        fpr = fp / max(1, fp + tn) * 100
        precision = tp / max(1, tp + fp) * 100
        print(f" labels                 : {labels_path} "
              f"({len(labels_by_key)} flows)")
        print(f" evaluation window      : {counted} packets scored (post-training), "
              f"{sum(post.values())} labelled "
              f"({sum(post.values()) / max(1, counted) * 100:.1f}% coverage)")
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
    if args.buckets:
        if not _print_drift_buckets(samples, args.buckets, counted):
            print("ERROR: the drift profile does not account for every scored "
                  "packet — its shape cannot be read as evidence.", file=sys.stderr)
            return 3
    print(f" pipeline counters      : processed={pipeline.total_processed} "
          f"blocked={pipeline.total_blocked}")
    print("=======================================================")
    return 0


def main() -> int:
    return asyncio.run(_evaluate(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
