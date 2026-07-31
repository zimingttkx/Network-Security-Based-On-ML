#!/usr/bin/env python3
"""
NIPS Attack Simulation — 大规模攻击检测效果测试。

生成 50,000+ 正常流量 + 30,000+ 攻击流量，送入完整检测流水线
(RuleEngine → Kitsune)，测量每种攻击类型的检出率和误报率。

用法:
    python scripts/attack_simulation.py           # 快速测试 (10万包)
    python scripts/attack_simulation.py --full    # 完整测试 (30万包)
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, ".")

from networksecurity.engine import DetectionPipeline, PacketInfo, Action, Verdict
from networksecurity.engine.kitsune.detector_adapter import KitsuneDetector
from networksecurity.engine.rule_engine import RuleEngine

# ═══════════════════════════════════════════════════════════════════════════
# 流量生成器 — 真实多样的正常/攻击流量模式
# ═══════════════════════════════════════════════════════════════════════════

# 正常用户 IP 池 (200 个内网 IP，模拟企业网络)
NORMAL_IPS = [
    f"192.168.{i}.{j}" for i in range(1, 15) for j in range(1, 20)
][:200]

# 攻击者 IP 池 (分散在全球的恶意 IP)
ATTACK_IPS = [
    f"{a}.{b}.{c}.{d}"
    for a in [45, 62, 91, 103, 121, 185, 198, 212]
    for b in range(10, 250, 30)
    for c in range(1, 200, 20)
    for d in range(1, 250, 30)
][:2000]

SERVER_IP = "10.0.0.1"
SERVER_PORTS = {
    "web": [80, 443, 8080, 8443],
    "ssh": [22],
    "db": [3306, 5432, 6379, 27017],
    "dns": [53],
    "mail": [25, 110, 143, 993, 995],
}


class TrafficGenerator:
    """生成真实的正常和攻击流量模式。"""

    _ip_idx = 0
    _attack_ip_idx = 0

    @staticmethod
    def normal_web(timestamp: float) -> PacketInfo:
        """正常 HTTP/HTTPS 浏览流量。"""
        src = random.choice(NORMAL_IPS)
        dst_port = random.choice(SERVER_PORTS["web"])
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(32768, 65535),
            dst_port=dst_port,
            protocol=6,
            packet_size=random.randint(60, 1460),
            tcp_flags=random.choice([0x02, 0x10, 0x18, 0x11]),
            ttl=random.randint(48, 128),
            timestamp=timestamp,
        )

    @staticmethod
    def normal_ssh(timestamp: float) -> PacketInfo:
        """正常 SSH 管理流量（小包、低频）。"""
        src = random.choice(NORMAL_IPS[:20])  # 少数管理员
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(32768, 65535),
            dst_port=22,
            protocol=6,
            packet_size=random.randint(40, 200),
            tcp_flags=random.choice([0x18, 0x10]),
            ttl=64,
            timestamp=timestamp,
        )

    @staticmethod
    def normal_db(timestamp: float) -> PacketInfo:
        """正常数据库查询流量。"""
        src = random.choice(NORMAL_IPS[:50])
        dst_port = random.choice(SERVER_PORTS["db"])
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(32768, 65535),
            dst_port=dst_port,
            protocol=6,
            packet_size=random.randint(64, 4096),
            tcp_flags=0x18,
            ttl=64,
            timestamp=timestamp,
        )

    @staticmethod
    def normal_dns(timestamp: float) -> PacketInfo:
        """正常 DNS 查询。"""
        src = random.choice(NORMAL_IPS)
        return PacketInfo(
            src_ip=src, dst_ip=random.choice(["8.8.8.8", "10.0.0.53"]),
            src_port=random.randint(32768, 65535),
            dst_port=53,
            protocol=17,
            packet_size=random.randint(40, 150),
            ttl=64,
            timestamp=timestamp,
        )

    # ── 攻击流量 ──────────────────────────────────────────────────────

    @staticmethod
    def syn_flood(timestamp: float) -> PacketInfo:
        """SYN Flood — 大量伪造 IP 发送 SYN 包。"""
        src = random.choice(ATTACK_IPS)
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(1, 65535),
            dst_port=80,
            protocol=6,
            packet_size=40,
            tcp_flags=0x02,  # SYN only
            ttl=random.randint(32, 255),
            timestamp=timestamp,
        )

    @staticmethod
    def port_scan(timestamp: float) -> PacketInfo:
        """Port Scan — 少数 IP 快速扫描大范围端口。"""
        src = random.choice(ATTACK_IPS[:50])
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(40000, 50000),
            dst_port=random.randint(1, 65535),
            protocol=6,
            packet_size=40,
            tcp_flags=0x02,
            ttl=random.randint(48, 128),
            timestamp=timestamp,
        )

    @staticmethod
    def udp_flood(timestamp: float) -> PacketInfo:
        """UDP Flood DDoS — 高频率大包 UDP。"""
        src = random.choice(ATTACK_IPS)
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(1, 65535),
            dst_port=53,
            protocol=17,
            packet_size=random.randint(200, 1500),
            ttl=random.randint(32, 255),
            timestamp=timestamp,
        )

    @staticmethod
    def icmp_flood(timestamp: float) -> PacketInfo:
        """ICMP Flood (Smurf-like) — 大量 ping。"""
        src = random.choice(ATTACK_IPS[:300])
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=0, dst_port=0,
            protocol=1,  # ICMP → 被 RuleEngine protocol filter 直接拦
            packet_size=random.randint(64, 1500),
            ttl=255,
            timestamp=timestamp,
        )

    @staticmethod
    def slowloris(timestamp: float) -> PacketInfo:
        """Slowloris 类攻击 — 不完整 HTTP 请求，小包维持连接。"""
        src = random.choice(ATTACK_IPS[:100])
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(32768, 65535),
            dst_port=80,
            protocol=6,
            packet_size=random.randint(20, 60),
            tcp_flags=0x18,
            ttl=128,
            timestamp=timestamp,
        )

    @staticmethod
    def brute_force_ssh(timestamp: float) -> PacketInfo:
        """SSH 暴力破解 — 高频小包到 22 端口。"""
        src = random.choice(ATTACK_IPS[:80])
        return PacketInfo(
            src_ip=src, dst_ip=SERVER_IP,
            src_port=random.randint(1024, 65535),
            dst_port=22,
            protocol=6,
            packet_size=random.randint(40, 80),
            tcp_flags=0x18,
            ttl=random.randint(48, 255),
            timestamp=timestamp,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 仿真运行器
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttackResult:
    name: str
    total: int = 0
    detected: int = 0
    detector_hits: dict = field(default_factory=lambda: defaultdict(int))


class SimulationRunner:
    """运行大规模攻击仿真，收集检测指标。"""

    def __init__(self, fast_mode: bool = False):
        # 快速模式：缩短 KitNET 训练周期
        fm_grace = 1000 if fast_mode else 5000
        ad_grace = 10000 if fast_mode else 50000

        self.pipeline = DetectionPipeline()
        self.kitsune = KitsuneDetector(
            max_autoencoder_size=10,
            threshold_percentile=99.0,
        )
        self.kitsune._kitsune.fm_grace = fm_grace
        self.kitsune._kitsune.ad_grace = ad_grace
        self.pipeline.add_detector(self.kitsune)

        # 预配置规则引擎
        self.pipeline.rule_engine.add_blacklist("45.33.32.156")

        self.normal_count = 0
        self.attack_results: dict[str, AttackResult] = {}
        self.fp_count = 0  # false positives
        self.tn_count = 0
        self._start_time = 0.0
        self.fast_mode = fast_mode

    async def generate_normal_traffic(self, count: int, ts_start: float = 0.0):
        """生成多样化正常流量。"""
        generators = [
            (TrafficGenerator.normal_web, 0.55),
            (TrafficGenerator.normal_ssh, 0.10),
            (TrafficGenerator.normal_db, 0.20),
            (TrafficGenerator.normal_dns, 0.15),
        ]

        for i in range(count):
            ts = ts_start + i * 0.001  # 1ms 间隔 → 1000 pkt/s
            gen = random.choices([g for g, _ in generators],
                                 weights=[w for _, w in generators])[0]
            pkt = gen(ts)
            verdict = await self.pipeline.process_packet(pkt)
            self.normal_count += 1

            if verdict.action == Action.BLOCK:
                self.fp_count += 1
            else:
                self.tn_count += 1

            if (i + 1) % 5000 == 0:
                elapsed = time.monotonic() - self._start_time
                rate = (i + 1) / max(0.001, elapsed)
                print(f"    正常流量: {i + 1}/{count} ({rate:.0f} pkt/s)")

    async def run_attack_phase(
        self, name: str, generator, count: int, ts_start: float, interval: float
    ):
        """运行一种攻击，记录检测结果。"""
        result = AttackResult(name=name)
        self.attack_results[name] = result

        for i in range(count):
            ts = ts_start + i * interval
            pkt = generator(ts)
            verdict = await self.pipeline.process_packet(pkt)
            result.total += 1

            if verdict.action == Action.BLOCK:
                result.detected += 1
                result.detector_hits[verdict.detector] += 1

        return result

    async def run_simulation(self, normal_packets: int, attack_scale: int):
        """运行完整仿真。

        Args:
            normal_packets: 正常流量包数 (≥10000 才能让 KitNET 训练)
            attack_scale: 每种攻击的包数基数
        """
        self._start_time = time.monotonic()

        # ═══ Phase 1: 正常流量训练 ═══
        print("\n" + "─" * 60)
        print("Phase 1: 正常流量 (KitNET 训练)")
        print("─" * 60)
        t0 = time.monotonic()
        await self.generate_normal_traffic(normal_packets)
        t1 = time.monotonic()
        normal_rate = normal_packets / max(0.001, t1 - t0)
        print(f"  ✓ 完成: {normal_packets} 包, {normal_rate:.0f} pkt/s")
        print(f"  KitNET 训练完成: {self.kitsune.is_ready}")
        print(f"  误报 (FPR): {self.fp_count}/{self.normal_count} "
              f"({self.fp_count/max(1,self.fp_count+self.tn_count)*100:.2f}%)")

        # ═══ Phase 2: 混合攻击 ═══
        attack_start_ts = normal_packets * 0.001 + 10.0

        attack_scenarios = [
            ("SYN Flood",    TrafficGenerator.syn_flood,        attack_scale * 3, 0.0001),  # 10000 pkt/s
            ("Port Scan",    TrafficGenerator.port_scan,        attack_scale * 2, 0.0002),  # 5000 pkt/s
            ("UDP DDoS",     TrafficGenerator.udp_flood,        attack_scale * 2, 0.0002),
            ("ICMP Flood",   TrafficGenerator.icmp_flood,       attack_scale,     0.0005),
            ("Slowloris",    TrafficGenerator.slowloris,        attack_scale,     0.001),
            ("SSH Brute",    TrafficGenerator.brute_force_ssh,  attack_scale,     0.001),
        ]

        attack_ts = attack_start_ts
        for name, gen, count, interval in attack_scenarios:
            print(f"\n─ Phase: {name} ({count} packets) ─")
            t0 = time.monotonic()
            await self.run_attack_phase(name, gen, count, attack_ts, interval)
            t1 = time.monotonic()
            rate = count / max(0.001, t1 - t0)
            result = self.attack_results[name]
            dr = result.detected / max(1, result.total) * 100
            print(f"  ✓ 完成: {count} 包, {rate:.0f} pkt/s")
            print(f"  检出: {result.detected}/{result.total} ({dr:.1f}%)")
            if result.detector_hits:
                hits = ", ".join(f"{k}: {v}" for k, v in result.detector_hits.items())
                print(f"  检测器: {hits}")
            attack_ts += count * interval + 5.0  # 间隔 5s

        # ═══ Phase 3: 混合流量 (正常+攻击交错) ═══
        print("\n" + "─" * 60)
        print("Phase 3: 混合流量 (正常 60% + 攻击 40%)")
        print("─" * 60)

        mixed_normal = 0
        mixed_attack = 0
        mixed_detected = 0
        mixed_fp = 0
        mixed_total = attack_scale * 3
        mixed_start_ts = attack_ts + 10.0

        t0 = time.monotonic()
        for i in range(mixed_total):
            ts = mixed_start_ts + i * 0.0005  # 2000 pkt/s

            if random.random() < 0.6:
                # 正常流量
                pkt = TrafficGenerator.normal_web(ts)
                verdict = await self.pipeline.process_packet(pkt)
                mixed_normal += 1
                if verdict.action == Action.BLOCK:
                    mixed_fp += 1
            else:
                # 随机攻击
                attack_type = random.choice([
                    TrafficGenerator.syn_flood,
                    TrafficGenerator.port_scan,
                    TrafficGenerator.udp_flood,
                ])
                pkt = attack_type(ts)
                verdict = await self.pipeline.process_packet(pkt)
                mixed_attack += 1
                if verdict.action == Action.BLOCK:
                    mixed_detected += 1

        t1 = time.monotonic()
        mixed_rate = mixed_total / max(0.001, t1 - t0)

        print(f"  ✓ 完成: {mixed_total} 包, {mixed_rate:.0f} pkt/s")
        print(f"  正常包: {mixed_normal}, 攻击包: {mixed_attack}")
        print(f"  攻击检出: {mixed_detected}/{mixed_attack} "
              f"({mixed_detected/max(1,mixed_attack)*100:.1f}%)")
        print(f"  误报: {mixed_fp}/{mixed_normal} "
              f"({mixed_fp/max(1,mixed_normal)*100:.2f}%)")

        return mixed_detected, mixed_attack, mixed_fp, mixed_normal

    def print_report(self):
        """输出完整的仿真报告。"""
        total_elapsed = time.monotonic() - self._start_time
        total_packets = self.pipeline.total_processed

        print("\n")
        print("╔" + "═" * 63 + "╗")
        print("║" + "  NIPS Attack Simulation — 检测效果报告".center(57) + "║")
        print("╠" + "═" * 63 + "╣")

        # 环境信息
        import platform
        print(f"║  环境: Python {sys.version.split()[0]} | "
              f"{platform.node()} | {platform.platform()[:20]:20s} ║")
        print(f"║  总包数: {total_packets:,} | "
              f"耗时: {total_elapsed:.0f}s | "
              f"平均: {total_packets/max(0.001,total_elapsed):.0f} pkt/s  ║")
        print(f"║  KitNET 训练完成: {str(self.kitsune.is_ready):5s}"
              f"                          ║")
        print("╠" + "═" * 63 + "╣")

        # 各攻击类型检测率
        print("║  攻击类型        总包数    检出数    检出率    主要检测器      ║")
        print("╠" + "─" * 63 + "╣")

        for name, result in self.attack_results.items():
            dr = result.detected / max(1, result.total) * 100
            main_detector = max(result.detector_hits, key=result.detector_hits.get) \
                if result.detector_hits else "N/A"
            bar = "█" * int(dr / 5) + ("░" * (20 - int(dr / 5)))
            print(f"║  {name:14s}  {result.total:>6,}   {result.detected:>6,}   "
                  f"{dr:>5.1f}%  {bar}  {main_detector:14s} ║")

        print("╠" + "═" * 63 + "╣")

        # 汇总指标
        total_attack = sum(r.total for r in self.attack_results.values())
        total_detected = sum(r.detected for r in self.attack_results.values())
        overall_dr = total_detected / max(1, total_attack) * 100
        overall_fpr = self.fp_count / max(1, self.fp_count + self.tn_count) * 100

        print(f"║  汇总                                                       ║")
        print(f"║    总攻击包: {total_attack:>8,}                                     ║")
        print(f"║    总检出:   {total_detected:>8,}  ({overall_dr:.1f}%)                          ║")
        print(f"║    总正常包: {self.normal_count:>8,}                                     ║")
        print(f"║    误报(FPR): {self.fp_count:>6,}  ({overall_fpr:.2f}%)                            ║")
        print(f"║    流水线:   RuleEngine → Kitsune(AfterImage+KitNET)         ║")

        print("╚" + "═" * 63 + "╝")

        # 检测器贡献分析
        detector_totals: dict[str, int] = defaultdict(int)
        for result in self.attack_results.values():
            for det, count in result.detector_hits.items():
                detector_totals[det] += count
        if detector_totals:
            print("\n  检测器贡献分布:")
            total_hits = sum(detector_totals.values())
            for det, count in sorted(detector_totals.items(),
                                     key=lambda x: x[1], reverse=True):
                pct = count / max(1, total_hits) * 100
                bar = "█" * int(pct / 2)
                print(f"    {det:20s}  {count:>6,} ({pct:5.1f}%)  {bar}")

        # 吞吐量分阶段统计
        print(f"\n  管道状态: {self.pipeline.status()}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="NIPS 大规模攻击仿真")
    parser.add_argument("--full", action="store_true",
                        help="完整模式 (300,000+ 包)")
    parser.add_argument("--normal", type=int, default=0,
                        help="正常流量包数 (覆盖默认值)")
    parser.add_argument("--scale", type=int, default=0,
                        help="攻击包基数 (覆盖默认值)")
    args = parser.parse_args()

    if args.full:
        normal_packets = 100_000
        attack_scale = 20_000
    else:
        normal_packets = args.normal or 30_000
        attack_scale = args.scale or 5_000

    total_attack = attack_scale * (3 + 2 + 2 + 1 + 1 + 1)  # sum of weights
    total_all = normal_packets + total_attack + attack_scale * 3  # + mixed
    print(f"\n{'='*65}")
    print(f"NIPS 攻击仿真")
    print(f"{'='*65}")
    print(f"  配置: {normal_packets:,} 正常 + ~{total_attack + attack_scale*3:,} 攻击 "
          f"= ~{total_all:,} 包")
    fast = not args.full
    print(f"  模式: {'快速 (短训练周期)' if fast else '完整 (标准训练周期)'}")
    print(f"  检测器: RuleEngine + Kitsune (AfterImage 115维 + KitNET)")

    runner = SimulationRunner(fast_mode=fast)
    await runner.run_simulation(normal_packets, attack_scale)
    runner.print_report()


if __name__ == "__main__":
    asyncio.run(main())
