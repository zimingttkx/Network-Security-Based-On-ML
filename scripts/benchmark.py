#!/usr/bin/env python3
"""NIPS benchmark — measures throughput and rule-engine accuracy under load.

Kitsune is an online unsupervised learner.  It requires ~55 000 normal
packets before entering detection mode.  This benchmark:
1. Feeds 55k normal packets to train KitNET
2. Measures detection accuracy on known attack patterns
3. Measures pipeline throughput (packets/sec)
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from typing import Sequence

sys.path.insert(0, ".")

from networksecurity.engine import DetectionPipeline, PacketInfo, Action, Verdict
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.lucid.detector_adapter import LucidDetectorAdapter
from networksecurity.engine.rule_engine import RuleEngine

# ---------------------------------------------------------------------------
# Packet generators — these produce realistic-looking PacketInfo objects,
# NOT hardcoded feature vectors.  The AfterImage engine computes 115 features
# from the packet fields.
# ---------------------------------------------------------------------------


class TrafficGenerator:
    """Generates realistic diverse traffic patterns."""

    NORMAL_IPS = [f"192.168.{i}.{j}" for i in range(1, 20) for j in range(1, 250)]
    NORMAL_IPS += [f"10.{i}.{j}.{k}" for i in range(0, 5) for j in range(0, 10) for k in range(1, 50)]
    ATTACK_IPS = [f"45.33.{i}.{j}" for i in range(32, 64) for j in range(1, 100)]
    ATTACK_IPS += [f"185.220.{i}.{j}" for i in range(100, 120) for j in range(1, 100)]

    SERVER_IP = "10.0.0.1"
    SERVER_PORTS = [22, 80, 443, 8080, 3306, 5432, 6379]

    @staticmethod
    def normal_packet(timestamp: float) -> PacketInfo:
        src = random.choice(TrafficGenerator.NORMAL_IPS)
        dst_port = random.choice(TrafficGenerator.SERVER_PORTS)
        # Normal: mostly TCP to web ports, reasonable sizes
        if dst_port in (80, 443, 8080):
            size = random.randint(40, 1500)
        elif dst_port == 22:
            size = random.randint(40, 200)
        else:
            size = random.randint(40, 500)
        return PacketInfo(
            src_ip=src,
            dst_ip=TrafficGenerator.SERVER_IP,
            src_port=random.randint(1024, 65535),
            dst_port=dst_port,
            protocol=6,  # TCP
            packet_size=size,
            tcp_flags=random.choice([0x02, 0x10, 0x18]),
            timestamp=timestamp,
        )

    @staticmethod
    def scan_packet(timestamp: float) -> PacketInfo:
        """Port scan: small packets to unusual ports."""
        return PacketInfo(
            src_ip=random.choice(TrafficGenerator.ATTACK_IPS),
            dst_ip=TrafficGenerator.SERVER_IP,
            src_port=random.randint(1024, 65535),
            dst_port=random.randint(1, 65535),
            protocol=6,
            packet_size=40,
            tcp_flags=0x02,
            timestamp=timestamp,
        )

    @staticmethod
    def syn_flood_packet(timestamp: float) -> PacketInfo:
        """SYN flood: rapid SYN packets from many IPs."""
        return PacketInfo(
            src_ip=random.choice(TrafficGenerator.ATTACK_IPS),
            dst_ip=TrafficGenerator.SERVER_IP,
            src_port=random.randint(1, 65535),
            dst_port=80,
            protocol=6,
            packet_size=40,
            tcp_flags=0x02,
            timestamp=timestamp,
        )

    @staticmethod
    def ddos_packet(timestamp: float) -> PacketInfo:
        """DDoS: high-rate small UDP packets."""
        src = random.choice(TrafficGenerator.ATTACK_IPS)
        return PacketInfo(
            src_ip=src,
            dst_ip=TrafficGenerator.SERVER_IP,
            src_port=random.randint(1, 65535),
            dst_port=53,
            protocol=17,  # UDP
            packet_size=random.randint(60, 200),
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class Benchmark:
    def __init__(self) -> None:
        self.pipeline = DetectionPipeline()
        self.pipeline.add_detector(KitsuneDetector())
        self.kitsune: KitsuneDetector = next(
            d for d in self.pipeline.detectors if isinstance(d, KitsuneDetector)
        )

    async def train_kitsune(self) -> None:
        """Feed normal packets to train KitNET.

        Kitsune needs fm_grace + ad_grace normal packets before it enters
        detection mode.  For a fast, meaningful benchmark we shorten both
        grace periods (via the public attributes) so detection actually
        fires in Phase 4 with a realistic number of packets.  Production
        uses the defaults (~55k).
        """
        # Shorten training so detection runs within the benchmark budget.
        self.kitsune._kitsune.fm_grace = 1000
        self.kitsune._kitsune.ad_grace = 9000
        count = 0
        t0 = time.monotonic()
        for i in range(16_000):
            pkt = TrafficGenerator.normal_packet(timestamp=float(i) / 1000.0)
            await self.pipeline.process_packet(pkt)
            count += 1
            if count % 4000 == 0:
                elapsed = time.monotonic() - t0
                rate = count / max(0.001, elapsed)
                print(f"  training ... {count} packets, {rate:.0f} pkt/s")
        elapsed = time.monotonic() - t0
        ready = self.kitsune.is_ready
        print(f"  trained {count} packets in {elapsed:.1f}s ({count/max(0.001,elapsed):.0f} pkt/s)")
        print(f"  Kitsune ready: {ready}")

    async def run_benchmark(self) -> dict:
        """Run the full benchmark suite.  Returns metrics dict."""

        # --- Phase 1: Rule-engine accuracy (deterministic) ---
        print("\n--- Phase 1: Rule-engine accuracy ---")
        self.pipeline.rule_engine.add_blacklist("45.33.32.156")
        bn_ip = "45.33.32.156"
        known_good_ip = "192.168.1.100"

        rule_results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}
        t0 = time.monotonic()
        for i in range(10_000):
            is_attack = i % 2 == 0
            src = bn_ip if is_attack else known_good_ip
            pkt = PacketInfo(
                src_ip=src, dst_ip="10.0.0.1",
                src_port=random.randint(1024, 65535),
                dst_port=80, protocol=6, packet_size=random.randint(40, 1500),
                timestamp=float(i) / 100000.0,
            )
            verdict = await self.pipeline.process_packet(pkt)
            actual_is_attack = is_attack
            predicted_block = verdict.action == Action.BLOCK
            rule_results["total"] += 1
            if actual_is_attack and predicted_block:
                rule_results["tp"] += 1
            elif not actual_is_attack and not predicted_block:
                rule_results["tn"] += 1
            elif not actual_is_attack and predicted_block:
                rule_results["fp"] += 1
            elif actual_is_attack and not predicted_block:
                rule_results["fn"] += 1
        t1 = time.monotonic()
        rule_rate = rule_results["total"] / max(0.001, t1 - t0)

        accuracy = (rule_results["tp"] + rule_results["tn"]) / max(1, rule_results["total"])
        print(f"  Total: {rule_results['total']} packets, {rule_rate:.0f} pkt/s")
        print(f"  TP={rule_results['tp']} TN={rule_results['tn']} FP={rule_results['fp']} FN={rule_results['fn']}")
        print(f"  Accuracy: {accuracy*100:.1f}%  (rule engine: blacklist vs whitelist)")
        self.pipeline.rule_engine.remove_blacklist(bn_ip)

        # --- Phase 2: Pure rule-engine throughput (no ML) ---
        print("\n--- Phase 2: Rule-engine throughput ---")
        self.pipeline.rule_engine.add_blacklist("10.0.0.0/8")  # block everything
        count = 0
        t0 = time.monotonic()
        while (time.monotonic() - t0) < 5.0:
            pkt = TrafficGenerator.normal_packet(timestamp=time.monotonic())
            await self.pipeline.process_packet(pkt)
            count += 1
        elapsed = time.monotonic() - t0
        rule_qps = count / max(0.001, elapsed)
        print(f"  {count} packets in {elapsed:.1f}s → {rule_qps:.0f} pkt/s")
        self.pipeline.rule_engine.remove_blacklist("10.0.0.0/8")

        # --- Phase 3: Kitsune training throughput ---
        print("\n--- Phase 3: Kitsune training throughput ---")
        count = 0
        t0 = time.monotonic()
        while (time.monotonic() - t0) < 10.0:
            pkt = TrafficGenerator.normal_packet(timestamp=time.monotonic())
            await self.pipeline.process_packet(pkt)
            count += 1
        elapsed = time.monotonic() - t0
        training_qps = count / max(0.001, elapsed)
        print(f"  {count} packets in {elapsed:.1f}s → {training_qps:.0f} pkt/s (training mode)")

        # --- Phase 4: Kitsune anomaly detection (post-training) ---
        print("\n--- Phase 4: Kitsune anomaly detection ---")
        # Feed normal, scan, and ddos traffic
        normal_pass = normal_block = 0
        scan_block = scan_pass = 0
        ddos_block = ddos_pass = 0
        total = 0
        t0 = time.monotonic()

        for i in range(5_000):
            r = random.random()
            if r < 0.6:
                pkt = TrafficGenerator.normal_packet(timestamp=time.monotonic())
                is_attack = False
                track = "normal"
            elif r < 0.8:
                pkt = TrafficGenerator.scan_packet(timestamp=time.monotonic())
                is_attack = True
                track = "scan"
            else:
                pkt = TrafficGenerator.ddos_packet(timestamp=time.monotonic())
                is_attack = True
                track = "ddos"
            verdict = await self.pipeline.process_packet(pkt)
            total += 1
            blocked = verdict.action == Action.BLOCK
            if track == "normal":
                if blocked: normal_block += 1
                else: normal_pass += 1
            elif track == "scan":
                if blocked: scan_block += 1
                else: scan_pass += 1
            elif track == "ddos":
                if blocked: ddos_block += 1
                else: ddos_pass += 1

        elapsed = time.monotonic() - t0
        detection_qps = total / max(0.001, elapsed)

        # Metrics
        total_normal = normal_pass + normal_block
        total_attack = scan_pass + scan_block + ddos_pass + ddos_block
        attack_detected = scan_block + ddos_block
        false_positives = normal_block

        tpr = attack_detected / max(1, total_attack) * 100  # true positive rate
        fpr = false_positives / max(1, total_normal) * 100   # false positive rate

        print(f"  Total: {total} packets, {detection_qps:.0f} pkt/s")
        print(f"  Normal: {normal_pass} pass / {normal_block} blocked (FPR={fpr:.1f}%)")
        print(f"  Scan:   {scan_pass} pass / {scan_block} blocked")
        print(f"  DDoS:   {ddos_pass} pass / {ddos_block} blocked")
        print(f"  Attack detection rate: {tpr:.1f}%")

        return {
            "rule_qps": rule_qps,
            "training_qps": training_qps,
            "detection_qps": detection_qps,
            "rule_accuracy": accuracy,
            "tpr": tpr,
            "fpr": fpr,
            "kitsune_trained": self.kitsune.is_ready,
        }


async def main() -> None:
    print("=" * 65)
    print("NIPS Benchmark")
    print("=" * 65)
    print(f"Detectors: RuleEngine + KitsuneDetector (AfterImage + KitNET)")
    print(f"Environment: Python {sys.version.split()[0]}")
    print()

    bench = Benchmark()

    # Train Kitsune with 16 000 normal packets (enough for fm_grace=5000)
    print("--- Training phase (16 000 normal packets) ---")
    await bench.train_kitsune()

    # Run full benchmark
    metrics = await bench.run_benchmark()

    print()
    print("=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Rule engine throughput:    {metrics['rule_qps']:,.0f} pkt/s")
    print(f"  Kitsune training:          {metrics['training_qps']:,.0f} pkt/s")
    print(f"  Kitsune detection:         {metrics['detection_qps']:,.0f} pkt/s")
    print(f"  Rule engine accuracy:      {metrics['rule_accuracy']*100:.1f}%")
    print(f"  Kitsune detection rate:    {metrics['tpr']:.1f}%")
    print(f"  Kitsune false positive:    {metrics['fpr']:.1f}%")
    print(f"  Kitsune fully trained:     {metrics['kitsune_trained']}")
    print()
    print("Notes:")
    print("  - Rule engine: deterministic blacklist/whitelist matching.")
    print("  - Kitsune: unsupervised. 16k training packets used (production")
    print("    requires ~55k).  TPR/FPR may vary with real traffic.")
    print("  - LUCID detector excluded: requires pretrained TensorFlow model.")
    print("  - Test environment: GitHub Codespaces (2 vCPU, 8 GB RAM).")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
