#!/usr/bin/env python3
"""Is the answer key on the wire? — the checks that keep it off, in one place.

A benchmark is only as good as its ground truth, and this one proved how
quietly that can fail: the builder drew attack flows' source addresses from one
documented block and normal flows' from another, so the label was recoverable
from the packet header — any source-address rule scored 100% for free — and the
evaluator recovered truth the same way, which made the measurement agree with
the reconstruction instead of measuring the detector.

Three things have to hold for the numbers to mean anything, and each is a check
here rather than a comment:

* **The label must not be predictable from the capture.**  The statistics run on
  addresses read back out of the pcap, not on the sidecar's own record of them:
  a sidecar that agrees with itself says nothing about the wire.
* **The sidecar must describe *this* capture.**  Every flow in it appears on the
  wire with the same endpoints and the same packet count, and every flow on the
  wire is in it.  A drifted answer key scores whatever it likes.
* **The evaluator must not invent truth.**  No sidecar, a sidecar that covers
  nothing, or an explicit ``--no-labels``: all three must end without rates
  rather than with the 0.0% an empty labelled set produces.

All of it is cheap — one pcap read, two statistics, three short evaluator runs —
so it gates pull requests (``verify_capture_truth.py``) and not just the nightly.
``verify_real_capture_quality.py`` imports it and refuses to spend half an hour
measuring on a capture whose truth it cannot trust.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_unsw_pcap import SRC, VICTIM
from capture_labels import flow_key, read_sidecar, sidecar_path, write_sidecar

ROOT = Path(__file__).resolve().parent.parent
PCAP = ROOT / "datasets/unsw-nb15/unsw_reconstructed.pcap"
SIDECAR = sidecar_path(PCAP)
BUILDER = ROOT / "scripts/build_unsw_pcap.py"
EVAL = ROOT / "scripts/evaluate_pcap.py"

# The convention the first evaluator used to recover truth from the wire.  It
# survives only as a canary: if it ever separates the classes again, the label
# is back in the packet header.
RETIRED_TRUTH_PREFIX = "175.45.176."
MIN_INDEPENDENCE_P = 0.01
MAX_RULE_ACCURACY_MARGIN = 0.05
SHUFFLES = 1000

# The evaluator probes ask whether it refuses to report rates, not whether it
# detects anything, so a token window at grace periods short enough to finish in
# seconds is enough.  Scored against the shipped 5k/50k this file would take
# minutes and prove the same thing.
PROBE_LIMIT = 200
PROBE_FM_GRACE = 10
PROBE_AD_GRACE = 10
# KitNET draws its autoencoder weights from the global RNG, so an unpinned
# probe's operating point is a property of the draw: over twelve seeds the
# high-percentile bucket run flags 0, 2 or 29 of the same 140 scored packets.
# Seed 0 is the point documented in check_buckets_are_portable; pinning it is
# what makes "the two operating points are distinguishable" a claim about the
# code rather than about this run's luck.
PROBE_SEED = 0


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT))


def _tail(run: subprocess.CompletedProcess, lines: int = 4) -> str:
    text = (run.stderr or run.stdout).strip().splitlines()
    return " / ".join(text[-lines:]) if text else f"exit {run.returncode}"


def wire_flows(capture: Path = PCAP) -> dict[str, dict]:
    """Per-flow endpoints and packet count, read back out of the capture itself.

    Deliberately independent of ``PcapLoader``: this is the check, so it has to
    see the frames as they are on the wire rather than as the pipeline's parser
    accepts them.  A frame the loader skips is a frame the evaluator never
    scores, and that shows up here as a packet count that disagrees with the
    sidecar.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    flows: dict[str, dict] = {}
    with PcapReader(str(capture)) as reader:
        for pkt in reader:
            ip = pkt.getlayer(IP)
            if ip is None:
                continue
            l4 = ip.getlayer(TCP) or ip.getlayer(UDP)
            if l4 is None:
                continue
            key = flow_key(int(ip.proto), ip.src, int(l4.sport), ip.dst, int(l4.dport))
            record = flows.setdefault(key, {"addresses": set(), "packets": 0})
            record["addresses"].update((ip.src, ip.dst))
            record["packets"] += 1
    return flows


def window_keys(limit: int, start: int = 0, capture: Path = PCAP) -> set[str]:
    """Flow keys present between packets ``start`` and ``limit``, read off the wire.

    A refusal probe has to land inside the window the evaluator actually scores;
    pick labels that never reach it and the "partial truth" case silently
    degenerates into the empty-label case it exists to tell apart from.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    keys: set[str] = set()
    with PcapReader(str(capture)) as reader:
        for i, pkt in enumerate(reader):
            if i >= limit:
                break
            if i < start:
                continue
            ip = pkt.getlayer(IP)
            if ip is None:
                continue
            l4 = ip.getlayer(TCP) or ip.getlayer(UDP)
            if l4 is None:
                continue
            keys.add(flow_key(int(ip.proto), ip.src, int(l4.sport), ip.dst,
                              int(l4.dport)))
    return keys


def check_alignment(records: list[dict], wire: dict[str, dict]) -> list[str]:
    """The answer key has to describe the capture it will be scored against."""
    sidecar_keys = {r["key"] for r in records}
    missing = sorted(sidecar_keys - set(wire))
    extra = sorted(set(wire) - sidecar_keys)
    print(f"answer key     : {len(records)} labelled flows, {len(wire)} flows on the wire")
    bad: list[str] = []
    if missing:
        bad.append(f"{len(missing)} labelled flows are not in the capture, "
                   f"first: {missing[0]}")
    if extra:
        bad.append(f"{len(extra)} capture flows carry no label, first: {extra[0]}")
    if bad:
        return bad

    by_key = {r["key"]: r for r in records}
    stray = [key for key in wire if wire[key]["packets"] != by_key[key]["packets"]]
    if stray:
        key = stray[0]
        bad.append(f"{len(stray)} flows carry a different packet count on the wire "
                   f"than in the sidecar, first: {key} "
                   f"(wire {wire[key]['packets']}, sidecar {by_key[key]['packets']})")
    odd = [key for key in wire
           if wire[key]["addresses"] != {by_key[key]["client"], VICTIM}]
    if odd:
        key = odd[0]
        bad.append(f"{len(odd)} flows do not run between their recorded client and "
                   f"the victim, first: {key} "
                   f"({sorted(wire[key]['addresses'])} on the wire vs "
                   f"{sorted({by_key[key]['client'], VICTIM})} in the sidecar)")
    return bad


def check_answer_key(records: list[dict], wire: dict[str, dict]) -> list[str]:
    """The label must not be recoverable from the packet header."""
    bad = check_alignment(records, wire)
    if bad:
        # Statistics over a partly-joined key would only add noise to the report.
        return bad

    clients = [next(a for a in wire[r["key"]]["addresses"] if a != VICTIM)
               for r in records]
    labels = [int(r["label"]) for r in records]
    stat, p = _address_independence(clients, labels)
    accuracy, base = _retired_rule_accuracy(clients, labels)
    canary = _canary_self_test()
    print(f"                 address independence: chi2={stat:.1f}, p={p:.3f}"
          f"   (canary self-test: {'ok' if not canary else 'BROKEN'})")
    print(f"                 retired '{RETIRED_TRUTH_PREFIX}* => attack' rule: "
          f"accuracy {accuracy:.3f} vs base rate {base:.3f}")
    bad += canary
    if p < MIN_INDEPENDENCE_P:
        bad.append(f"the source address predicts the label (chi2={stat:.1f}, "
                   f"p={p:.3f} < {MIN_INDEPENDENCE_P}) — the answer key is on the wire")
    if accuracy > base + MAX_RULE_ACCURACY_MARGIN:
        bad.append(f"the retired prefix rule still scores {accuracy:.3f} against a "
                   f"{base:.3f} base rate")
    return bad


def check_evaluator_refuses() -> list[str]:
    """Truth is an input: the evaluator must never infer or invent it."""
    bad: list[str] = []
    probes = []

    missing = _run([sys.executable, str(EVAL), "--labels", "does-not-exist.json",
                    "--limit", "1"], timeout=300)
    missing_out = missing.stdout + missing.stderr
    refused_missing = missing.returncode != 0
    if not refused_missing:
        bad.append("the evaluator ran with a missing label file instead of refusing")
    elif "does-not-exist.json" not in missing_out:
        bad.append("the evaluator refused a missing label file without naming it")
    if "detection rate" in missing_out or "false positive rate" in missing_out:
        bad.append("the evaluator printed rates with no labels to print them from")
    probes.append(f"missing sidecar {'refused' if refused_missing else 'ACCEPTED'}")

    plain = _run([sys.executable, str(EVAL), "--no-labels",
                  "--fm-grace", str(PROBE_FM_GRACE), "--ad-grace", str(PROBE_AD_GRACE),
                  "--limit", str(PROBE_LIMIT)], timeout=900)
    out = plain.stdout
    before = len(bad)
    if plain.returncode != 0:
        bad.append("the unlabelled run crashed")
    for field in ("detection rate", "false positive rate", "precision"):
        if field in out:
            bad.append(f"the unlabelled run printed a {field} with no ground truth")
    if not re.search(r"pcap packets processed\s*: [1-9]", out):
        bad.append("the unlabelled run processed nothing")
    probes.append(f"--no-labels {'printed no rates' if len(bad) == before else 'LEAKED RATES'}")

    # A sidecar describing none of this capture used to score 0.0% false
    # positives and 0.0% detection: an empty labelled set produces exactly the
    # numbers a perfect detector produces.  Refusing is the only safe answer.
    with tempfile.TemporaryDirectory() as tmp:
        drifted = Path(tmp) / "drifted.labels.json"
        write_sidecar(drifted, capture=PCAP, source="capture_truth probe", seed=0,
                      flows=[{"key": "tcp|192.0.2.1:1234|192.0.2.2:80", "label": 1,
                              "attack_cat": "Probe", "client": "192.0.2.1",
                              "packets": 1}])
        run = _run([sys.executable, str(EVAL), "--labels", str(drifted),
                    "--fm-grace", str(PROBE_FM_GRACE), "--ad-grace", str(PROBE_AD_GRACE),
                    "--limit", str(PROBE_LIMIT)], timeout=900)
        out = run.stdout + run.stderr
        refused = run.returncode != 0
        if not refused:
            bad.append("the evaluator scored a sidecar that describes none of the "
                       "capture instead of refusing it")
        elif "detection rate" in out or "false positive rate" in out:
            bad.append("the evaluator printed rates over an empty labelled set")
        probes.append(f"drifted sidecar {'refused' if refused else 'SCORED'}")

    # A *partially* matching sidecar is the same trap in disguise.  Five attack
    # packets and no normal ones used to print "false positive rate 0.0%" over an
    # empty class — a number no reader can tell from a clean run.  The refusal
    # has to name the class that is missing, not just fail.
    with tempfile.TemporaryDirectory() as tmp:
        _, flows = read_sidecar(SIDECAR)
        # Packets after the probe's warm-up (10 + 10 + 1) are the ones scored, so
        # the labels have to be found there or the probe proves nothing.
        in_window = window_keys(limit=PROBE_LIMIT, start=30)
        attack_only = [f for f in flows.values()
                       if f["label"] and f["key"] in in_window]
        if not attack_only:
            bad.append("the partial-truth probe found no attack flow inside its own "
                       "scoring window — it would have degenerated into the "
                       "empty-label case")
        else:
            partial = Path(tmp) / "attack-only.labels.json"
            write_sidecar(partial, capture=PCAP, source="capture_truth probe", seed=0,
                          flows=attack_only)
            half = _run([sys.executable, str(EVAL), "--labels", str(partial),
                         "--fm-grace", str(PROBE_FM_GRACE),
                         "--ad-grace", str(PROBE_AD_GRACE),
                         "--limit", str(PROBE_LIMIT)], timeout=900)
            out = half.stdout + half.stderr
            before = len(bad)
            if half.returncode == 0:
                bad.append("the evaluator scored a window that labels no normal "
                           "traffic — one of the two rates had no denominator")
            elif "normal" not in out:
                bad.append("the partial-truth refusal did not name the missing class: "
                           f"{_tail(half)}")
            for field in ("detection rate", "false positive rate", "precision"):
                if field in out:
                    bad.append(f"the partial-truth run printed a {field} over a window "
                               f"with no normal packets to measure it against")
            # The refusal is the last word an operator gets, so its own numbers
            # have to add up: labelled + unmatched must equal what was scored.
            # Run-wide `unmatched` used to be quoted here, mixing in warm-up
            # packets and printing "5 + 178" for a window of 179.
            sums = re.search(r"\((\d+) packets scored, (\d+) attack \+ (\d+) normal "
                             r"labelled, (\d+) unmatched in the window", out)
            if not sums:
                bad.append("the partial-truth refusal no longer quotes numbers that "
                           f"can be checked: {_tail(half)}")
            elif int(sums.group(2)) + int(sums.group(3)) + int(sums.group(4)) \
                    != int(sums.group(1)):
                bad.append("the refusal's own numbers do not add up: "
                           f"scored={sums.group(1)} attack={sums.group(2)} "
                           f"normal={sums.group(3)} unmatched={sums.group(4)}")
            probes.append(f"attack-only sidecar "
                          f"{'refused' if len(bad) == before else 'SCORED'}")

    print(f"evaluator      : {', '.join(probes)}")
    return bad


def _chi2(clients: list[str], labels: list[int]) -> float:
    """Pearson chi-square for independence between source address and label."""
    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for client, label in zip(clients, labels, strict=True):
        table[client][label] += 1
    share = sum(labels) / len(labels)
    stat = 0.0
    for normal, attack in table.values():
        size = normal + attack
        for observed, expected in ((normal, size * (1 - share)), (attack, size * share)):
            if expected > 0:
                stat += (observed - expected) ** 2 / expected
    return stat


def _address_independence(clients: list[str], labels: list[int],
                          shuffles: int = SHUFFLES,
                          seed: int = 7) -> tuple[float, float]:
    """(chi-square, permutation p-value) for "the source address predicts the label".

    The null is exactly "address carries no information about the label", and it
    is calibrated by shuffling labels across the same address partition — no
    distribution tables, and no assumption about how many flows each address
    happens to send.  A capture whose addresses encode the label scores orders
    of magnitude above every shuffle and lands at the p-value floor.
    """
    observed = _chi2(clients, labels)
    rng = random.Random(seed)
    shuffled = list(labels)
    at_least = 0
    for _ in range(shuffles):
        rng.shuffle(shuffled)
        if _chi2(clients, shuffled) >= observed:
            at_least += 1
    return observed, (at_least + 1) / (shuffles + 1)


def _retired_rule_accuracy(clients: list[str], labels: list[int]) -> tuple[float, float]:
    """Accuracy of the retired prefix rule, against the majority-class base rate."""
    share = sum(labels) / len(labels)
    hits = sum(1 for client, label in zip(clients, labels, strict=True)
               if client.startswith(RETIRED_TRUTH_PREFIX) == bool(label))
    return hits / len(labels), max(share, 1 - share)


def _canary_self_test() -> list[str]:
    """The canary must fire on a partition where the address does encode the label.

    A check that cannot fail is not a check.  This rebuilds the retired
    convention — attacks in one block, normal traffic in the other — and
    requires the canary to reject it, so a later edit that loosens the
    thresholds fails here instead of silently passing the real capture.
    """
    clients = ([f"175.45.176.{i % 200 + 2}" for i in range(300)]
               + [f"147.46.0.{i % 200 + 2}" for i in range(300)])
    labels = [1] * 300 + [0] * 300
    _, p = _address_independence(clients, labels)
    accuracy, base = _retired_rule_accuracy(clients, labels)
    bad: list[str] = []
    if p >= MIN_INDEPENDENCE_P:
        bad.append(f"the canary accepted a label-split partition (p={p:.3f})")
    if accuracy <= base + MAX_RULE_ACCURACY_MARGIN:
        bad.append(f"the retired rule failed to separate a label-split partition "
                   f"({accuracy:.3f} vs base {base:.3f})")
    return bad


def ensure_capture() -> list[str]:
    """Rebuild the gitignored capture and its sidecar if either is missing or stale.

    The builder is seeded, so a rebuild reproduces the previous capture and
    sidecar byte for byte and the numbers stay comparable (verified by hashing
    both files either side of a rebuild).  A builder that emits no capture, or
    emits one without its answer key, is a failure to report rather than a
    reason to skip: everything below is about that answer key existing and
    describing the capture.

    "Stale" matters as much as "missing": with the capture on disk, an edit to
    the builder would otherwise be checked against the *previous* build — a
    check that cannot fail, in the one place a developer tries it first.  In CI
    neither file exists, so this is a straight build.
    """
    if PCAP.exists() and SIDECAR.exists():
        built_at = min(PCAP.stat().st_mtime, SIDECAR.stat().st_mtime)
        sources = [BUILDER.stat().st_mtime] + [p.stat().st_mtime
                                               for p in (ROOT / SRC,) if p.exists()]
        if built_at >= max(sources):
            return []
    built = _run([sys.executable, str(BUILDER)], timeout=900)
    if not PCAP.exists() or not SIDECAR.exists():
        missing = "the capture" if not PCAP.exists() else "its answer key"
        return [f"the builder did not produce {missing}: {_tail(built)}"]
    wrote = (built.stdout or built.stderr).strip().splitlines()
    print(f"rebuilt {PCAP.name}: {wrote[-1] if wrote else 'no output'}", flush=True)
    return []


def check_buckets_are_portable() -> list[str]:
    """--buckets adds a reading of the same run; it must not move a field.

    The nightly gate parses this summary by first regex match, so a bucket block
    that reused one of those labels would silently change what the gate reads —
    a green pipeline measuring a different number than before.  Same 200 packets
    both ways: every gated label appears exactly once, and the default summary
    stays exactly as silent about buckets as it always was.
    """
    base = [sys.executable, str(EVAL), "--fm-grace", str(PROBE_FM_GRACE),
            "--ad-grace", str(PROBE_AD_GRACE), "--limit", str(PROBE_LIMIT),
            "--seed", str(PROBE_SEED)]
    plain = _run(base, timeout=900)
    bucketed = _run(base + ["--buckets", "7"], timeout=900)
    if plain.returncode or bucketed.returncode:
        return [f"the evaluator exited {plain.returncode}/{bucketed.returncode} "
                f"with/without --buckets: {_tail(bucketed) or _tail(plain)}"]

    bad: list[str] = []
    for label in ("pcap packets processed", "Kitsune trained", "unmatched packets",
                  "TP=", "false positive rate", "detection rate (TPR)",
                  "block reasons"):
        seen = bucketed.stdout.count(label)
        if seen != 1:
            bad.append(f"the bucketed summary contains {label!r} {seen} times; the "
                       f"gate's first match would read an ambiguous summary")
    if "drift profile" not in bucketed.stdout:
        bad.append("--buckets printed no drift profile")
    if "drift profile" in plain.stdout:
        bad.append("the default summary grew a drift profile nobody asked for")
    # End-to-end, not just the printer: the profile the run actually printed has
    # to account for every packet that run scored.  Otherwise a series collected
    # from part of the window still reads as a clean drift profile, and this is
    # the reading a calibration decision gets made on.
    recon = re.search(r"\((\d+)/(\d+) scored packets accounted for\)", bucketed.stdout)
    if not recon:
        bad.append("--buckets printed no reconciliation counts in its drift profile "
                   f"header: {_tail(bucketed)}")
    elif int(recon.group(2)) == 0 or int(recon.group(1)) != int(recon.group(2)):
        bad.append(f"the drift profile accounts for {recon.group(1)} of "
                   f"{recon.group(2)} scored packets")

    # The same reading at the other end of the operating point.  At the probe's
    # default threshold every scored packet is flagged, so a series filtered down
    # to the blocked ones would still reconcile 179/179 — a coincidence that
    # hides the defect.  A wider calibration window at the top percentile flags
    # 29 of 140 at the pinned seed, and there the same two numbers become a real
    # test.
    def _totals(text: str) -> tuple[int, int]:
        pairs = re.findall(r"bucket\s+\d+\s+t\+\s*[\d.]+s\s+(\d+)/(\d+)", text)
        return sum(int(f) for f, _ in pairs), sum(int(t) for _, t in pairs)

    sparse = _run(base + ["--buckets", "7", "--calibration-packets", "40",
                          "--threshold-percentile", "100"], timeout=900)
    if sparse.returncode:
        bad.append(f"the high-percentile --buckets run exited {sparse.returncode}: "
                   f"{_tail(sparse)}")
    recon2 = re.search(r"\((\d+)/(\d+) scored packets accounted for\)", sparse.stdout)
    if not recon2:
        bad.append("the high-percentile run printed no reconciliation counts in its "
                   f"drift profile header: {_tail(sparse)}")
    else:
        counted2 = int(recon2.group(2))
        flagged2, bucketed2 = _totals(sparse.stdout)
        if counted2 == 0 or int(recon2.group(1)) != counted2:
            bad.append(f"the high-percentile drift profile accounts for "
                       f"{recon2.group(1)} of {counted2} scored packets")
        if bucketed2 != counted2:
            bad.append(f"the high-percentile buckets hold {bucketed2} of {counted2} "
                       f"scored packets")
        # If both runs flag everything, the reconciliation cannot tell a
        # filtered series from a complete one and this check is decoration.
        if flagged2 >= counted2:
            bad.append(f"the high-percentile run still flagged every packet "
                       f"({flagged2}/{counted2}) — the two probe operating points "
                       f"are indistinguishable, so a series filtered to blocked "
                       f"packets would reconcile by coincidence")
    if not bad:
        print("buckets        : --buckets leaves every gated field unambiguous")
    return bad


def check_bucket_shape_is_honest() -> list[str]:
    """The drift profile has to account for every packet the run scored.

    Three ways this reading can lie without raising anything: a record whose
    timestamp falls outside the (first, last) pair lands in no bucket at all; a
    series collected from only part of the window (blocked packets, say) prints
    a self-consistent profile of a different run; and an out-of-range percentile
    is only refused by numpy after the capture has been trained on.  All three
    are checked here instead of being watched for.
    """
    import contextlib
    import io

    from evaluate_pcap import _print_drift_buckets

    def _bucketed(text: str) -> tuple[int, int]:
        pairs = re.findall(r"bucket\s+\d+\s+t\+\s*[\d.]+s\s+(\d+)/(\d+)", text)
        return sum(int(f) for f, _ in pairs), sum(int(t) for _, t in pairs)

    bad: list[str] = []
    # Non-monotonic, and wide enough that a first/last span still leaves the
    # time branch: the two out-of-order records sit below samples[0] and above
    # samples[-1], which is exactly what a (first, last) span drops.
    shuffled = [(10.0, False), (20.0, True), (5.0, True), (40.0, False), (30.0, True)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok_shuffled = _print_drift_buckets(shuffled, 4, len(shuffled))
    text = buf.getvalue()
    flagged, counted = _bucketed(text)
    want_flagged = sum(1 for _, f in shuffled if f)
    if not ok_shuffled:
        bad.append("a complete drift series was reported as not accounting for "
                   f"its packets: {text.strip().splitlines()[1] if len(text.splitlines()) > 1 else text}")
    if counted != len(shuffled):
        bad.append(f"the drift profile bucketed {counted} of {len(shuffled)} packets; "
                   f"out-of-order records are silently dropped")
    if "BUCKETING BUG" in text:
        bad.append("the bucket reconciliation line fired on a complete input: "
                   + next(line for line in text.splitlines() if "BUCKETING BUG" in line))
    if flagged != want_flagged:
        bad.append(f"the drift profile reports {flagged} flagged of {counted}, the "
                   f"input holds {want_flagged} — bucketing moved a verdict")

    ordered = [(float(i), i % 7 == 0) for i in range(60)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok_ordered = _print_drift_buckets(ordered, 5, len(ordered))
    _, counted = _bucketed(buf.getvalue())
    if not ok_ordered or counted != len(ordered):
        bad.append(f"a monotonic stream bucketed {counted} of {len(ordered)} packets")

    # The reconciliation has to be against the run's own scored count, not
    # against the list handed to the printer: a series that quietly kept only
    # blocked packets is self-consistent and describes a different run.
    partial = [(float(i), True) for i in range(20)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok_partial = _print_drift_buckets(partial, 4, 50)
    text = buf.getvalue()
    _, counted = _bucketed(text)
    if ok_partial:
        bad.append("a drift series holding 20 of 50 scored packets was reported as "
                   "complete — the printer reconciles against its own list, not the run")
    if counted != len(partial):
        bad.append(f"the truncated series bucketed {counted} of {len(partial)}")
    if "BUCKETING BUG" not in text:
        bad.append("the truncated series printed no reconciliation line")
    if "20/50 scored packets accounted for" not in text:
        bad.append("the truncated series did not name both counts in its header")

    # The range is refused before the capture is opened, so a typo costs a
    # message rather than half an hour of training.
    for value in ("101", "-1", "nan"):
        run = _run([sys.executable, str(EVAL), "--pcap", "no-such-capture.pcap",
                    "--no-labels", "--threshold-percentile", value], timeout=300)
        out = run.stdout + run.stderr
        if run.returncode == 0:
            bad.append(f"--threshold-percentile {value} was accepted")
        elif "threshold-percentile" not in out:
            bad.append(f"--threshold-percentile {value} was refused without naming "
                       f"the flag: {_tail(run)}")
        elif "no-such-capture" in out:
            bad.append(f"--threshold-percentile {value} failed only after the "
                       f"evaluator went looking for the capture")
    if not bad:
        print("bucket shape   : every scored packet lands in exactly one bucket, and "
              "a short series is refused; bad percentiles refused up front")
    return bad


def run_truth_checks() -> list[str]:
    """Every cheap truth check, in one place.  Empty list means all passed."""
    bad = ensure_capture()
    if bad:
        return bad
    _, by_key = read_sidecar(SIDECAR)
    records = list(by_key.values())
    bad = check_answer_key(records, wire_flows())
    bad += check_evaluator_refuses()
    bad += check_buckets_are_portable()
    bad += check_bucket_shape_is_honest()
    meta = json.loads(SIDECAR.read_text(encoding="utf-8"))
    print(f"capture        : {meta['capture']} (builder seed {meta['seed']}, "
          f"source {Path(meta['source']).name})")
    return bad
