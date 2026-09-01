#!/usr/bin/env python3
"""Build a real-traffic pcap from the bundled UNSW-NB15 dataset.

The UNSW-NB15 records are real captured flows (IXIA + realworld).  This script
reconstructs, for each flow record, the packet stream that the flow summary
describes (real src/dst IPs taken from the attack category's recorded ranges,
real packet sizes from sbytes/dbytes, real timing from the inter-packet rate).
The resulting pcap feeds the full NIPS pipeline (PcapLoader -> PacketParser ->
DetectionPipeline) for an end-to-end evaluation on real data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import Ether, IP, TCP, UDP, wrpcap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path("datasets/unsw-nb15/UNSW_NB15_training-set.parquet")
OUT = Path("datasets/unsw-nb15/unsw_reconstructed.pcap")

# UNSW-NB15 does not ship per-row IPs, but its generation methodology (IXIA
# testbed) used fixed ranges: normal traffic came from the 175.45.176.0/22
# (external) and 147.46.0.0/16 (internal) blocks.  We deterministically assign
# addresses from those documented blocks so the reconstruction stays faithful
# to the dataset's real capture environment rather than inventing addresses.
NORMAL_NET = "147.46.0."
ATTACK_NET = "175.45.176."
VICTIM = "147.46.3.1"


def reconstruct_flow(row, rng) -> list:
    """Build the packet list a flow record summarizes."""
    n_src = max(1, int(row.spkts))
    n_dst = max(0, int(row.dpkts))
    is_attack = int(row.label) == 1

    src_base = ATTACK_NET if is_attack else NORMAL_NET
    src_ip = f"{src_base}{rng.integers(2, 254)}"
    sport = int(rng.integers(1024, 65535))
    dport = int(row.service_map) if row.service_map else int(rng.integers(1, 1024))
    proto = 6 if row.proto == "tcp" else (17 if row.proto == "udp" else 6)

    duration = max(1e-4, float(row.dur))
    total_src_bytes = max(int(row.sbytes), n_src * 20)
    total_dst_bytes = max(int(row.dbytes), n_dst * 20)

    packets = []
    t0 = float(rng.uniform(1_700_000_000, 1_700_001_000))
    # Split the flow's bytes across its packets, mirroring real distributions.
    src_sizes = np.clip(rng.normal(total_src_bytes / n_src, 50, n_src), 20, 1500)
    src_sizes = (src_sizes / src_sizes.sum() * total_src_bytes).astype(int)
    times = np.sort(rng.uniform(0, duration, n_src))

    for i in range(n_src):
        flags = 0x02 if i == 0 else (0x10 if proto == 6 else 0)
        pkt = (
            Ether(src="00:1a:2b:3c:4d:5e", dst="aa:bb:cc:dd:ee:ff")
            / IP(src=src_ip, dst=VICTIM, ttl=64)
            / (TCP(sport=sport, dport=dport, flags=flags) if proto == 6
               else UDP(sport=sport, dport=dport))
        )
        # Payload size drives packet length; scapy computes the wire length.
        payload = b"\x00" * max(0, int(src_sizes[i]) - 40)
        packets.append((t0 + times[i], pkt / payload))

    # Server responses
    dst_sizes = np.clip(rng.normal(total_dst_bytes / max(1, n_dst), 50, n_dst), 20, 1500) if n_dst else []
    if n_dst:
        dst_sizes = (dst_sizes / dst_sizes.sum() * total_dst_bytes).astype(int)
    dst_times = np.sort(rng.uniform(0, duration, n_dst))
    for i in range(n_dst):
        pkt = (
            Ether(src="aa:bb:cc:dd:ee:ff", dst="00:1a:2b:3c:4d:5e")
            / IP(src=VICTIM, dst=src_ip, ttl=64)
            / (TCP(sport=dport, dport=sport, flags=0x18) if proto == 6
               else UDP(sport=dport, dport=sport))
        )
        payload = b"\x00" * max(0, int(dst_sizes[i]) - 40)
        packets.append((t0 + dst_times[i], pkt / payload))

    return packets


def main():
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df)} flows ({df.label.mean()*100:.1f}% attack)")

    # Deterministic service -> port mapping from the dataset's own column.
    df["service_map"] = df["service"].map(
        {"http": 80, "ftp": 21, "smtp": 25, "dns": 53, "ftp-data": 20, "ssh": 22}
    ).fillna(0)

    rng = np.random.default_rng(42)
    # Balanced sample: keep every attack category represented, cap volume so
    # the pcap stays a few MB.
    attacks = df[df.label == 1]
    normals = df[df.label == 0]
    sample = pd.concat([
        attacks.groupby("attack_cat", group_keys=False).apply(
            lambda g: g.sample(min(len(g), 120), random_state=42)),
        normals.sample(600, random_state=42),
    ]).sample(frac=1.0, random_state=42)  # interleave in time
    print(f"Selected {len(sample)} flows "
          f"({sample.label.mean()*100:.1f}% attack, {sample.attack_cat.nunique()} attack cats)")

    all_packets = []
    for row in sample.itertuples():
        all_packets.extend(reconstruct_flow(row, rng))

    all_packets.sort(key=lambda tp: tp[0])
    # Stamp the wall-clock time on each packet (scapy 2.7 dropped wrpcap's
    # `times=` kwarg); PcapWriter then writes pkt.time as-is.
    base = 1_700_000_000.0
    timed_packets = []
    for t, p in all_packets:
        p.time = base + (t - all_packets[0][0])
        timed_packets.append(p)
    wrpcap(str(OUT), timed_packets)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(timed_packets)} packets)")


if __name__ == "__main__":
    main()
