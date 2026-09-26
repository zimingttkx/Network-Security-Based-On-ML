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


def run_truth_checks() -> list[str]:
    """Every cheap truth check, in one place.  Empty list means all passed."""
    bad = ensure_capture()
    if bad:
        return bad
    _, by_key = read_sidecar(SIDECAR)
    records = list(by_key.values())
    bad = check_answer_key(records, wire_flows())
    bad += check_evaluator_refuses()
    meta = json.loads(SIDECAR.read_text(encoding="utf-8"))
    print(f"capture        : {meta['capture']} (builder seed {meta['seed']}, "
          f"source {Path(meta['source']).name})")
    return bad
