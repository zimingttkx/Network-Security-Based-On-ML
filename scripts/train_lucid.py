#!/usr/bin/env python3
"""Train the LUCID DDoS detector and write the model that engine.lucid.model_path expects.

Until now the config invited you to "set a path to a trained Keras model" while
nothing in the repo could produce one, so LUCID was an advertised feature that
nobody could switch on.  This is that missing entry point.

Two stages, deliberately separated:

    --inspect   label and window the packets, print the class balance, write
                nothing.  Needs no TensorFlow, and answers the question that
                actually bites people: did I name the right attacker IPs?  A
                one-sided label set trains a model that predicts one class and
                looks healthy while doing it.
    (default)   do that, then fit the CNN and save it.  Requires TensorFlow:
                pip install -e ".[lucid]"

Packet source is a real pcap (tcpdump/scapy capture, or the one
build_unsw_pcap.py reconstructs from the bundled dataset).  Labels come from
addresses you supply: a flow window counts as an attack when a majority of its
packets involve one of the attacker or victim IPs on either side, which is
LUCID's own convention.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _split_addresses(values: list[str]) -> list[str]:
    """Accept --attackers a,b and repeated --attackers a --attackers b."""
    out: list[str] = []
    for value in values:
        path = Path(value)
        if len(value) == 1 and not value.isdigit() and path.exists():
            # A file of addresses, one per line: capture reports and firewall
            # logs are lists, not command-line material.
            out += [line.strip() for line in path.read_text().splitlines() if line.strip()]
        else:
            out += [part.strip() for part in value.split(",") if part.strip()]
    return out


async def _load_packets(pcap_path: str) -> list[dict]:
    from networksecurity.data.pcap_loader import PcapLoader

    packets: list[dict] = []
    loader = PcapLoader()
    async for entry in loader.load(pcap_path):
        if entry is None:
            continue
        packets.append(entry)
    return packets


def _summarise(packets: list[dict], attackers: list[str], victims: list[str],
               time_window: float, packets_per_flow: int) -> int:
    """Report the training set the labels imply, without needing TensorFlow."""
    from networksecurity.engine.lucid.dataset_parser import LucidDatasetParser

    parser = LucidDatasetParser(time_window=time_window, packets_per_flow=packets_per_flow)
    X, y = parser.build_samples(packets, attackers=attackers, victims=victims)
    attacks = int(np.count_nonzero(y))
    print(f"  packets read            : {len(packets)}")
    print(f"  complete {packets_per_flow}-packet windows : {len(X)}"
          f"  (shape {tuple(X.shape[1:])})")
    print(f"  windows labelled attack : {attacks}")
    print(f"  windows labelled benign : {len(X) - attacks}")
    print(f"  windows expired early   : {parser.expired_flows}")
    if not attackers and not victims:
        print("  WARNING: neither --attackers nor --victims given — every window "
              "is labelled benign and training cannot proceed.")
        return 1
    if len(X) == 0:
        print("  ERROR: no window reached "
              f"{packets_per_flow} packets inside {time_window}s. A capture that "
              "is too short, too sparse, or missing link-layer timing produces no "
              "samples; widen --time-window or capture more traffic.")
        return 1
    if attacks == 0 or attacks == len(X):
        print("  ERROR: labels are one-sided. Check the addresses you passed: with "
              "a victim listed, every flow toward it is counted as attack traffic, "
              "which is LUCID's convention but not what you want for background "
              "connections.")
        return 1
    print("  label set looks usable"
          f" (ratio {attacks / len(X):.2%} attack).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pcap", required=True, help="capture file to train from")
    ap.add_argument("--attackers", action="append", default=[],
                    help="attacker IPs (comma separated, or a file of one per line); "
                         "matched exactly, CIDR notation is not supported")
    ap.add_argument("--victims", action="append", default=[],
                    help="victim IPs (comma separated, or a file); note that every "
                         "flow toward a victim counts as attack traffic")
    ap.add_argument("--out", default="models/lucid_cnn.h5", help="where to save the model")
    ap.add_argument("--time-window", type=float, default=10.0)
    ap.add_argument("--packets-per-flow", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--validation-split", type=float, default=0.2)
    ap.add_argument("--inspect", action="store_true",
                    help="label and window the packets, then exit (no TensorFlow needed)")
    args = ap.parse_args()

    if args.time_window <= 0 or args.packets_per_flow < 2:
        print("ERROR: --time-window must be > 0 and --packets-per-flow >= 2", file=sys.stderr)
        return 2

    pcap = Path(args.pcap)
    if not pcap.exists():
        print(f"ERROR: no such pcap: {pcap}", file=sys.stderr)
        return 2

    attackers = _split_addresses(args.attackers)
    victims = _split_addresses(args.victims)
    print(f"Reading {pcap} ...")
    packets = asyncio.run(_load_packets(str(pcap)))
    if not packets:
        print("ERROR: the pcap yielded no parseable IPv4 TCP/UDP packets.", file=sys.stderr)
        return 1
    print(f"  attackers: {len(attackers)}   victims: {len(victims)}")

    status = _summarise(packets, attackers, victims, args.time_window, args.packets_per_flow)
    if args.inspect or status:
        return status

    try:
        from networksecurity.engine.lucid.detector import LucidDetector
    except ImportError as exc:
        print(f"ERROR: LUCID needs TensorFlow ({exc}). Install it with: "
              'pip install -e ".[lucid]"', file=sys.stderr)
        return 1

    detector = LucidDetector(time_window=args.time_window,
                             packets_per_flow=args.packets_per_flow)
    print(f"Training (epochs={args.epochs}) ...")
    try:
        history = detector.train_from_packets(packets, epochs=args.epochs,
                                              validation_split=args.validation_split,
                                              attackers=attackers, victims=victims)
    except ImportError as exc:
        print(f"ERROR: TensorFlow is required to fit the CNN ({exc}). Install it "
              'with: pip install -e ".[lucid]"', file=sys.stderr)
        return 1
    if isinstance(history, dict):
        for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
            value = history.get(key)
            if value:
                print(f"  {key}: {value[-1] if isinstance(value, list) else value}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not detector.save(str(out)):
        print(f"ERROR: failed to write {out}", file=sys.stderr)
        return 1
    print(f"Saved {out}")
    print("Enable it by setting in config/config.yaml:")
    print(f"  engine:\n    lucid:\n      model_path: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
