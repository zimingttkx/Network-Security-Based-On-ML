#!/usr/bin/env python3
"""Cross-validation: 3 bugs exposed by the live-interception session.

Bug A  loopback DNS stub 127.0.0.53 was permanently DROPped live (host DNS
       self-DoS).  Fix: IptablesManager refuses to block loopback sources,
       and the NIPS chain ACCEPTs everything arriving on `lo` so local
       packets never reach the detection pipeline.

Bug B  5 kernel DROP rules existed at runtime while shutdown saved
       rules.json with an empty blacklist (state lived only in
       IptablesManager._blocked).  Fix: Interceptor mirrors every block
       into pipeline.rule_engine.add_blacklist, so save_rules() persists it
       and the rule engine short-circuits later packets from that source.

Bug C  Interceptor.setup() called setup_nfqueue() with no argument: kernel
       always redirected to queue 0 while NFQueueHandler listened on the
       configured queue_num — any nfqueue_num != 0 freezes all traffic.
       Fix: setup() passes self._queue_num explicitly.

Exit 0 = all fixes verified; exit 1 = at least one check failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import inspect
import os
import tempfile

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict
from networksecurity.interception.iptables import IptablesManager
from networksecurity.interception.interceptor import Interceptor

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    if not cond:
        ok = False
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


class Blocker(BaseDetector):
    async def process_packet(self, packet: PacketInfo) -> Verdict:
        return Verdict(action=Action.BLOCK, confidence=0.99,
                       threat_level=ThreatLevel.HIGH,
                       reason="anomaly", detector=self.name)


class StubIptables:
    """Records block_ip calls; no kernel involvement."""

    def __init__(self):
        self.blocked: list[str] = []
        self.setup_queue: int | None = None

    def setup_nfqueue(self, queue_num: int = 0) -> None:
        self.setup_queue = queue_num

    def block_ip(self, ip: str) -> None:
        self.blocked.append(ip)

    def unblock_ip(self, ip: str) -> None:
        try:
            self.blocked.remove(ip)
        except ValueError:
            pass

    def cleanup_all(self) -> None:
        self.blocked.clear()

    def _chain_exists(self, chain: str) -> bool:
        return True


# ---------------------------------------------------------------- Bug A
print("=" * 60)
print("Bug A: loopback sources must never be blocked")
print("=" * 60)
ipt = IptablesManager(safe_ips=["127.0.0.1", "::1"])  # shipped config.yaml
ipt._nfqueue_rules_added = True  # simulate an active session (not teardown)
inserted: list[list[str]] = []
ipt._run = lambda *args, **kw: (inserted.append(list(args)), "")[1]  # type: ignore[method-assign]
ipt._chain_exists = staticmethod(lambda chain: True)  # type: ignore[method-assign]
ipt._rule_exists = staticmethod(lambda *args: False)  # type: ignore[method-assign]

ipt.block_ip("127.0.0.53")   # systemd-resolved stub — the live incident
ipt.block_ip("127.0.0.1")
ipt.block_ip("203.0.113.66")  # genuine remote attacker
check("A1: loopback stub 127.0.0.53 refused (no DROP inserted)",
      not any("127.0.0.53" in a for a in inserted),
      "iptables calls: %s" % [a for a in inserted if "DROP" in a])
check("A2: normal loopback 127.0.0.1 also refused",
      not any("127.0.0.1" in a for a in inserted))
check("A3: remote IP 203.0.113.66 still blocked",
      any("203.0.113.66" in a for a in inserted),
      "iptables calls: %s" % [a for a in inserted if "DROP" in a])

src = inspect.getsource(IptablesManager.setup_nfqueue)
check("A4: NIPS chain ACCEPTs traffic arriving on lo (loopback never queued)",
      '"-i", "lo"' in src or "-i lo" in src)

# ---------------------------------------------------------------- Bug B
print()
print("=" * 60)
print("Bug B: interceptor blocks must persist via rule_engine")
print("=" * 60)
rule_engine = RuleEngine()
pipeline = DetectionPipeline()
pipeline.set_rule_engine(rule_engine)
pipeline.add_detector(Blocker())

inter = Interceptor(pipeline, queue_num=0)
inter._iptables = StubIptables()  # type: ignore[assignment]

pkt = PacketInfo(src_ip="203.0.113.66", dst_ip="10.0.0.1", src_port=4444,
                 dst_port=80, protocol=6, packet_size=520, timestamp=1.0)
asyncio.run(inter._handle(pkt, {"timed_out": False}))

check("B1: interceptor recorded the kernel-side block",
      "203.0.113.66" in inter._iptables.blocked,
      "StubIptables.blocked = %s" % inter._iptables.blocked)
check("B2: block mirrored into rule_engine blacklist",
      "203.0.113.66" in rule_engine.get_blacklist(),
      "rule_engine.get_blacklist() = %s" % rule_engine.get_blacklist())

tmp = Path(tempfile.mkdtemp()) / "rules.json"
rule_engine.save_rules(tmp)
saved = tmp.read_text()
check("B3: persisted rules.json keeps the block across restart",
      "203.0.113.66" in saved,
      "rules.json = %s" % saved.strip().replace("\n", " "))

# A timeout verdict must still NOT commit a block anywhere.
asyncio.run(inter._handle(
    PacketInfo(src_ip="198.51.100.9", dst_ip="10.0.0.1", src_port=4444,
               dst_port=80, protocol=6, packet_size=520, timestamp=1.1),
    {"timed_out": True}))
check("B4: timed-out verdict commits no block (no kernel rule, no blacklist)",
      "198.51.100.9" not in inter._iptables.blocked
      and "198.51.100.9" not in rule_engine.get_blacklist())

# ---------------------------------------------------------------- Bug C
print()
print("=" * 60)
print("Bug C: kernel redirect queue must match the userspace listener")
print("=" * 60)
inter2 = Interceptor(pipeline, queue_num=7)  # config.yaml nfqueue_num: 7
stub2 = StubIptables()
inter2._iptables = stub2  # type: ignore[assignment]

_real_geteuid = os.geteuid
os.geteuid = lambda: 0  # simulate root for setup()
try:
    inter2.setup()
finally:
    os.geteuid = _real_geteuid
    inter2.stop()  # join the detection loop thread, clean state

check("C1: setup() redirects the kernel to the configured queue (7)",
      stub2.setup_queue == 7, "setup_nfqueue received queue %s" % stub2.setup_queue)
check("C2: userspace listener bound to the same queue (7)",
      inter2._nfqueue._queue_num == 7)

print()
print("=" * 60)
print("RESULT:", "ALL FIXES VERIFIED" if ok else "FAILURES PRESENT (see FAIL lines)")
print("=" * 60)
sys.exit(0 if ok else 1)
