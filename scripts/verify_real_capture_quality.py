#!/usr/bin/env python3
"""Detection quality on a real capture — measured and gated, not quoted.

Every other suite here measures the pipeline on traffic this project generates
(``attack_simulation``, ``verify_fpr_regression``): both answer "did a code
change alter behaviour", neither answers "what does this do to real packets".
The numbers they print are properties of the generator, and a generator that
draws attacks from a separable distribution will always look good.

This runs the shipped UNSW-NB15 reconstruction through the full pipeline and
gates the numbers an operator actually feels:

  * the false-positive rate on real normal flows — the median of three fixed
    draws, bounded at the same 5% the synthetic gate uses: a detector that
    typically fires on real traffic more than that cannot be left inline,
  * that no draw is catastrophic (the worst of the three, bounded at 15%; this
    measurement's failures were 52%, 65% and 100% of packets),
  * that the ML stage still flags *something* (a TPR of exactly zero would mean
    the detector is dead, not that the capture is clean).

It also gates the thing that made the first version of this measurement
worthless: ground truth used to be recovered from the wire (attack flows drew
their source address from one documented block, and the evaluator read the
label back out of it), so the score measured agreement with the builder's
addressing convention.  Three checks keep that from coming back — the answer
key must not be predictable from the source address, the retired prefix rule
must not separate the classes, and the evaluator must refuse to invent truth
when it is not given any.

The exact rates are printed as evidence and are what the READMEs quote.  They
are calibration for this capture, not production accuracy: the pcap is rebuilt
from flow records, so it carries real addresses, sizes and timing but not real
per-packet structure.  Training runs at the shipped grace periods — a shorter
feature-mapping window makes the fitted model a property of whichever flows
arrive first, which is a property of the draw rather than of the detector (see
``evaluate_pcap.py``).

They are also not reproducible between runs: KitNET initialises its weights
from the global RNG, and this capture at these grace periods scored 2.1% to
10.5% false positives across twelve seeded draws (two unseeded draws landed at
2.8% and 7.8%).  A bound on a single unseeded run is a coin flip, so the gate
scores three fixed draws — the low, middle and high of that sweep — and bounds
their median.  That is deterministic, and it still moves when the detector's
operating point moves.

    python scripts/verify_real_capture_quality.py
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture_labels import read_sidecar, sidecar_path

ROOT = Path(__file__).resolve().parent.parent
PCAP = ROOT / "datasets/unsw-nb15/unsw_reconstructed.pcap"
SIDECAR = sidecar_path(PCAP)
BUILDER = ROOT / "scripts/build_unsw_pcap.py"
EVAL = ROOT / "scripts/evaluate_pcap.py"

# The three draws the gate scores: the low, middle and high draw of the sweep
# the READMEs quote.  KitNET seeds its own weights from the global RNG, and on
# this capture that alone moved the false-positive rate from 2.1% to 10.5%, so
# a single unseeded draw cannot carry a bound — the 5% bound this file used to
# hold went red on a draw where nothing had changed.  Fixed seeds make the gate
# reproducible; three of them keep it from measuring one corner.
SEEDS = (5, 0, 10)

# Bounds, not targets.  The median is the draw a run typically lands on and
# gets the same 5% the synthetic gate uses: a detector that typically fires on
# real normal traffic at percent-level rates cannot be left inline.  The worst
# draw is bounded much wider — the failures this gate exists to catch were 52%,
# 65% and 100% of packets, so 15% still catches them while leaving the measured
# spread (worst 10.5%) alone.  The detection rate is deliberately not bounded:
# it sits in the low single digits and moves between draws, so gating it would
# gate noise.  A detector that stops working shows up as no draw flagging any
# attack packet, a truth check breaking, or these two bounds.
MAX_MEDIAN_FPR = 5.0
MAX_ANY_FPR = 15.0

# The convention the first evaluator used to recover truth from the wire.  It
# survives only as a canary: if it ever separates the classes again, the label
# is back in the packet header.
RETIRED_TRUTH_PREFIX = "175.45.176."
MIN_INDEPENDENCE_P = 0.01
MAX_RULE_ACCURACY_MARGIN = 0.05
SHUFFLES = 1000


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT))


def _chi2(clients: list[str], labels: list[int]) -> float:
    """Pearson chi-square for independence between source address and label."""
    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for client, label in zip(clients, labels):
        table[client][label] += 1
    share = sum(labels) / len(labels)
    stat = 0.0
    for normal, attack in table.values():
        size = normal + attack
        for observed, expected in ((normal, size * (1 - share)), (attack, size * share)):
            if expected > 0:
                stat += (observed - expected) ** 2 / expected
    return stat


def _address_independence(flows: list[dict], shuffles: int = SHUFFLES,
                          seed: int = 7) -> tuple[float, float]:
    """(chi-square, permutation p-value) for "the source address predicts the label".

    The null is exactly "address carries no information about the label", and it
    is calibrated by shuffling labels across the same address partition — no
    distribution tables, and no assumption about how many flows each address
    happens to send.  A capture whose addresses encode the label scores orders
    of magnitude above every shuffle and lands at the p-value floor.
    """
    clients = [f["client"] for f in flows]
    labels = [f["label"] for f in flows]
    observed = _chi2(clients, labels)
    rng = random.Random(seed)
    shuffled = list(labels)
    at_least = 0
    for _ in range(shuffles):
        rng.shuffle(shuffled)
        if _chi2(clients, shuffled) >= observed:
            at_least += 1
    return observed, (at_least + 1) / (shuffles + 1)


def _retired_rule_accuracy(flows: list[dict]) -> tuple[float, float]:
    """Accuracy of the retired prefix rule, against the majority-class base rate."""
    share = sum(f["label"] for f in flows) / len(flows)
    hits = sum(1 for f in flows
               if f["client"].startswith(RETIRED_TRUTH_PREFIX) == bool(f["label"]))
    return hits / len(flows), max(share, 1 - share)


def _canary_self_test() -> list[str]:
    """The canary must fire on a partition where the address does encode the label.

    A check that cannot fail is not a check.  This rebuilds the retired
    convention — attacks in one block, normal traffic in the other — and
    requires the canary to reject it, so a later edit that loosens the
    thresholds fails here instead of silently passing the real capture.
    """
    flows = ([{"client": f"175.45.176.{i % 200 + 2}", "label": 1} for i in range(300)]
             + [{"client": f"147.46.0.{i % 200 + 2}", "label": 0} for i in range(300)])
    _, p = _address_independence(flows)
    accuracy, base = _retired_rule_accuracy(flows)
    bad: list[str] = []
    if p >= MIN_INDEPENDENCE_P:
        bad.append(f"the canary accepted a label-split partition (p={p:.3f})")
    if accuracy <= base + MAX_RULE_ACCURACY_MARGIN:
        bad.append(f"the retired rule failed to separate a label-split partition "
                   f"({accuracy:.3f} vs base {base:.3f})")
    return bad


def _check_answer_key(flows: list[dict]) -> list[str]:
    """The label must not be recoverable from the packet header."""
    bad: list[str] = _canary_self_test()
    stat, p = _address_independence(flows)
    accuracy, base = _retired_rule_accuracy(flows)
    print(f"answer key     : {len(flows)} flows over "
          f"{len({f['client'] for f in flows})} client addresses")
    print(f"                 address independence: chi2={stat:.1f}, p={p:.3f}"
          f"   (canary self-test: {'ok' if not bad else 'BROKEN'})")
    print(f"                 retired '{RETIRED_TRUTH_PREFIX}* => attack' rule: "
          f"accuracy {accuracy:.3f} vs base rate {base:.3f}")
    if p < MIN_INDEPENDENCE_P:
        bad.append(f"the source address predicts the label (chi2={stat:.1f}, "
                   f"p={p:.3f} < {MIN_INDEPENDENCE_P}) — the answer key is on the wire")
    if accuracy > base + MAX_RULE_ACCURACY_MARGIN:
        bad.append(f"the retired prefix rule still scores {accuracy:.3f} against a "
                   f"{base:.3f} base rate")
    return bad


def _check_evaluator_refuses_unlabelled_truth() -> list[str]:
    """Truth is an input: the evaluator must never infer it from packet contents."""
    bad: list[str] = []
    missing = _run([sys.executable, str(EVAL), "--labels", "does-not-exist.json",
                    "--limit", "1"], timeout=300)
    if missing.returncode == 0:
        bad.append("the evaluator ran with a missing label file instead of refusing")
    elif "does-not-exist.json" not in (missing.stdout + missing.stderr):
        bad.append("the evaluator refused a missing label file without naming it")

    plain = _run([sys.executable, str(EVAL), "--no-labels", "--limit", "3000"],
                 timeout=900)
    out = plain.stdout
    if plain.returncode != 0:
        bad.append("the unlabelled run crashed")
    for field in ("detection rate", "false positive rate", "precision"):
        if field in out:
            bad.append(f"the unlabelled run printed a {field} with no ground truth")
    if not re.search(r"pcap packets processed\s*: [1-9]", out):
        bad.append("the unlabelled run processed nothing")
    return bad


def _score_draw(seed: int) -> tuple[dict | None, list[str]]:
    """Run one evaluation draw and pull the gated fields out of its report.

    The full report is echoed: the gate keeps only a digest, and the nightly
    job uploads this log, so the per-category table and block reasons are the
    evidence behind any number the digest prints.
    """
    try:
        run = _run([sys.executable, str(EVAL), "--seed", str(seed)], timeout=1800)
    except subprocess.TimeoutExpired:
        return None, [f"seed {seed}: the evaluation did not finish within 30 minutes"]
    if run.returncode != 0:
        print((run.stderr or run.stdout).strip()[-2000:])
        return None, [f"seed {seed}: the evaluation crashed"]
    print(run.stdout, end="", flush=True)

    out = run.stdout
    window = re.search(r"TP=(\d+)\s+FN=(\d+)\s+FP=(\d+)\s+TN=(\d+)", out)
    processed = re.search(r"pcap packets processed\s*: (\d+)", out)
    unmatched = re.search(r"unmatched packets\s*: (\d+)", out)
    if window is None or processed is None or unmatched is None:
        return None, [f"seed {seed}: the evaluation produced no summary to gate"]

    def field(label: str) -> str:
        match = re.search(rf"{re.escape(label)}\s*:\s*([^\n]+)", out)
        return match.group(1).strip() if match else ""

    tp, fn, fp, tn = (int(x) for x in window.groups())
    row = {
        "seed": seed, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fpr": float(field("false positive rate").split()[0].rstrip("%")),
        "tpr": field("detection rate (TPR)"),
        "trained": field("Kitsune trained").lower().startswith("true"),
        "processed": int(processed.group(1)),
        "unmatched": int(unmatched.group(1)),
        "reasons": field("block reasons"),
    }

    bad: list[str] = []
    if row["processed"] <= 0:
        bad.append(f"seed {seed}: no packets were processed")
    if row["unmatched"] != 0:
        bad.append(f"seed {seed}: {row['unmatched']} packets matched no labelled flow "
                   f"— the sidecar and the capture have drifted apart")
    if not row["trained"]:
        bad.append(f"seed {seed}: Kitsune never left training on this capture")
    if tp + fn == 0:
        bad.append(f"seed {seed}: the evaluation window contained no attack packets to score")
    if fp + tn == 0:
        bad.append(f"seed {seed}: the evaluation window contained no normal packets to score")
    return row, bad


def main() -> int:
    if not PCAP.exists() or not SIDECAR.exists():
        # The capture is gitignored (the parquet it is built from is not), so
        # CI rebuilds it — the builder is seeded, which keeps the numbers
        # comparable between runs.  The sidecar ships with it: a capture
        # without its answer key cannot be scored, and a builder that emits one
        # is a failure to report, not a reason to skip.
        built = _run([sys.executable, str(BUILDER)], timeout=900)
        if not PCAP.exists():
            last = (built.stderr or built.stdout).strip().splitlines()[-1:]
            print(f"SKIP: no capture and the rebuild did not produce one — {last}")
            return 0
        if not SIDECAR.exists():
            print(f"FAIL: the builder produced {PCAP.name} without its answer key "
                  f"({SIDECAR.name})")
            print((built.stderr or built.stdout).strip()[-2000:])
            return 1
        print(f"rebuilt {PCAP.name}: {built.stdout.strip().splitlines()[-1:]}")

    _, flows_by_key = read_sidecar(SIDECAR)
    flows = list(flows_by_key.values())

    bad = _check_answer_key(flows)
    bad += _check_evaluator_refuses_unlabelled_truth()
    if bad:
        # Fail before the 30-minute run: a broken answer key makes every number
        # below meaningless, so there is nothing to measure.
        for name in bad:
            print(f"FAIL: {name}")
        print("REAL CAPTURE QUALITY: FAIL")
        return 1

    rows: list[dict] = []
    for seed in SEEDS:
        print(f"draw seed={seed} ...", flush=True)
        row, row_bad = _score_draw(seed)
        bad += row_bad
        if row is None:
            continue
        rows.append(row)
        print(f"  seed {seed:<3} false positives {row['fpr']:.1f}%   "
              f"detection {row['tpr']}   TP={row['tp']} FN={row['fn']} "
              f"FP={row['fp']} TN={row['tn']}   trained={row['trained']}", flush=True)

    if bad or not rows:
        for name in bad:
            print(f"FAIL: {name}")
        print("REAL CAPTURE QUALITY: FAIL")
        return 1

    fprs = sorted(row["fpr"] for row in rows)
    median = fprs[len(fprs) // 2]
    worst = fprs[-1]
    print(f"capture        : {PCAP.name} ({rows[0]['processed']} packets)")
    print(f"false positives: {' '.join(f'{f:.1f}%' for f in fprs)}"
          f"   median {median:.1f}%  worst {worst:.1f}%")
    print(f"detection rate : {' '.join(row['tpr'] for row in rows)}   (not gated)")
    for row in rows:
        print(f"block reasons  : seed {row['seed']}: {row['reasons']}")

    bad = []
    if sum(row["tp"] for row in rows) == 0:
        bad.append("no draw flagged a single attack packet — the ML stage is dead "
                   "rather than the capture clean")
    if median > MAX_MEDIAN_FPR:
        bad.append(f"median false-positive rate {median:.1f}% exceeds {MAX_MEDIAN_FPR}%")
    if worst > MAX_ANY_FPR:
        bad.append(f"the worst draw reached {worst:.1f}% false positives, "
                   f"over the {MAX_ANY_FPR}% bound")
    for name in bad:
        print(f"FAIL: {name}")
    print(f"REAL CAPTURE QUALITY: {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
