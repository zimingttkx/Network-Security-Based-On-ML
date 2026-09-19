#!/usr/bin/env python3
"""Cross-validation: the bugs exposed by the live-interception session.

Bug A  loopback DNS stub 127.0.0.53 was permanently DROPped live (host DNS
       self-DoS).  Fix: IptablesManager refuses to block loopback sources,
       and the NIPS chain ACCEPTs everything arriving on `lo` so local
       packets never reach the detection pipeline.

Bug B  5 kernel DROP rules existed at runtime while shutdown saved
       rules.json with an empty blacklist (state lived only in
       IptablesManager._blocked).  Fix: Interceptor mirrors every block
       into the rule engine — a temp ban into the ephemeral tier (matched,
       never saved, lifted by the sweeper), a perm ban promoted into the
       persistent tier and written by save_rules() — so nothing the kernel
       enforces is invisible to restart or to the API.

Bug C  Interceptor.setup() called setup_nfqueue() with no argument: kernel
       always redirected to queue 0 while NFQueueHandler listened on the
       configured queue_num — any nfqueue_num != 0 freezes all traffic.
       Fix: setup() passes self._queue_num explicitly.

Bug H  Escalation committed the blacklist mirror before knowing whether
       the kernel DROP landed, so a refused or failed iptables insert left
       a ban the API reported and rules.json persisted while nothing
       enforced it.  Fix: block_ip() returns whether the rule is really
       installed; _enforce_ban() gates every mirror layer on that result
       and queues the IP in _pending_enforce for the sweeper to retry.

Exit 0 = all fixes verified; exit 1 = at least one check failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import inspect
import os
import tempfile

from networksecurity.engine.block_policy import BlockPolicy
from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict
from networksecurity.interception.iptables import IptablesManager, blockable
import networksecurity.interception.interceptor as interceptor_mod
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
    """Records block_ip calls; no kernel involvement.

    Applies the real blockable() criteria and the real return contract so
    the Bug B mirror path is exercised against the same loopback/safe-ips
    refusals — and the same "did the kernel rule really change state"
    signal — that the iptables manager produces.
    """

    def __init__(self, safe_ips=None):
        self.blocked: list[str] = []
        self.setup_queue: int | None = None
        self._safe_ips = safe_ips or []

    def setup_nfqueue(self, queue_num: int = 0, intercept_icmp: bool = False) -> None:
        self.setup_queue = queue_num
        self.setup_icmp = intercept_icmp

    def block_ip(self, ip: str) -> bool:
        if not blockable(ip, self._safe_ips):
            return False
        if ip not in self.blocked:
            self.blocked.append(ip)
        return True

    def unblock_ip(self, ip: str) -> bool:
        try:
            self.blocked.remove(ip)
        except ValueError:
            return False
        return True

    def is_blockable(self, ip: str) -> bool:
        return blockable(ip, self._safe_ips)

    def cleanup_all(self) -> None:
        self.blocked.clear()

    def _chain_exists(self, chain: str) -> bool:
        return True


class RefusingIptables(StubIptables):
    """Fails the kernel insert for ``fail_ip`` while ``refusing`` is set.

    Stands in for iptables rejecting a rule (table locked, chain rebuilt by
    another process) on a source that IS blockable — the case where the
    interceptor must not commit any mirror layer.
    """

    def __init__(self, fail_ip: str):
        super().__init__()
        self.fail_ip = fail_ip
        self.refusing = True

    def block_ip(self, ip: str) -> bool:
        if ip == self.fail_ip and self.refusing:
            return False
        return super().block_ip(ip)


# ---------------------------------------------------------------- Bug A
print("=" * 60)
print("Bug A: loopback sources must never be blocked")
print("=" * 60)
ipt = IptablesManager(safe_ips=["127.0.0.1", "::1"])  # shipped config.yaml
ipt._nfqueue_rules_added = True  # simulate an active session (not teardown)
inserted: list[list[str]] = []
ipt._run = lambda *args, **kw: (inserted.append(list(args)), "")[1]  # type: ignore[method-assign]
ipt._chain_exists = staticmethod(lambda chain, **kw: True)  # type: ignore[method-assign]
ipt._rule_exists = staticmethod(lambda *args, **kw: False)  # type: ignore[method-assign]

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

# Graduated enforcement (engine/block_policy.py, commit b78aaa6): a single
# BLOCK only counts a strike — the kernel DROP + blacklist mirror happen on
# threshold crossing.  threshold=1 keeps this check to one packet.
policy = BlockPolicy(strikes_threshold=1, temp_ban_count_to_perm=99)
inter = Interceptor(pipeline, queue_num=0, block_policy=policy)
inter._iptables = StubIptables()  # type: ignore[assignment]

pkt = PacketInfo(src_ip="203.0.113.66", dst_ip="10.0.0.1", src_port=4444,
                 dst_port=80, protocol=6, packet_size=520, timestamp=1.0)
NO_DEADLINE = float("inf")
EXPIRED_DEADLINE = -1.0
asyncio.run(inter._handle(pkt, NO_DEADLINE))

check("B1: interceptor recorded the kernel-side block",
      "203.0.113.66" in inter._iptables.blocked,
      f"StubIptables.blocked = {inter._iptables.blocked}")
check("B2: temp ban mirrored into the rule engine's EPHEMERAL blacklist",
      "203.0.113.66" in rule_engine.get_ephemeral_blacklist(),
      f"get_ephemeral_blacklist() = {rule_engine.get_ephemeral_blacklist()}")

tmp = Path(tempfile.mkdtemp()) / "rules.json"
rule_engine.save_rules(tmp)
saved = tmp.read_text()
check("B3: a TEMP ban is deliberately not persisted (reversible by design)",
      "203.0.113.66" not in saved,
      f"rules.json = {saved.strip()}")

# The original Bug B guarantee — a kernel DROP is never invisible to
# rules.json — now lives on the permanent path, which promotes the mirror
# into the savable tier and writes the file in the same step.
PERM = "203.0.113.77"
tmp_perm = Path(tempfile.mkdtemp()) / "rules.json"
real_rules_file = interceptor_mod.RULES_FILE
interceptor_mod.RULES_FILE = tmp_perm
try:
    asyncio.run(inter._enforce_ban(PERM, True))
finally:
    interceptor_mod.RULES_FILE = real_rules_file
saved_perm = tmp_perm.read_text() if tmp_perm.exists() else ""
check("B3b: a PERM ban reaches the kernel, the persistent tier and rules.json",
      PERM in inter._iptables.blocked
      and PERM in rule_engine.get_blacklist()
      and PERM in saved_perm
      and PERM not in rule_engine.get_ephemeral_blacklist(),
      f"blocked={inter._iptables.blocked} "
      f"blacklist={rule_engine.get_blacklist()} "
      f"rules.json={saved_perm.strip()}")

# A verdict that resolves after the inline-drop deadline must still NOT
# commit a block anywhere: the packet was already dropped, and the verdict
# may have been overtaken by a timeout.
asyncio.run(inter._handle(
    PacketInfo(src_ip="198.51.100.9", dst_ip="10.0.0.1", src_port=4444,
               dst_port=80, protocol=6, packet_size=520, timestamp=1.1),
    EXPIRED_DEADLINE))
check("B4: expired-deadline verdict commits no block (no kernel rule, no mirror)",
      "198.51.100.9" not in inter._iptables.blocked
      and "198.51.100.9" not in rule_engine.get_blacklist()
      and "198.51.100.9" not in rule_engine.get_ephemeral_blacklist())

# ---------------------------------------------------------------- Bug H
print()
print("=" * 60)
print("Bug H: a refused kernel DROP must not leave a mirror behind")
print("=" * 60)
ATTACKER_H = "198.51.100.88"
engine_h = RuleEngine()
pipeline_h = DetectionPipeline()
pipeline_h.set_rule_engine(engine_h)
pipeline_h.add_detector(Blocker())
policy_h = BlockPolicy(strikes_threshold=1, temp_ban_count_to_perm=99)
inter_h = Interceptor(pipeline_h, queue_num=0, block_policy=policy_h)
stub_h = RefusingIptables(ATTACKER_H)
inter_h._iptables = stub_h  # type: ignore[assignment]

asyncio.run(inter_h._handle(
    PacketInfo(src_ip=ATTACKER_H, dst_ip="10.0.0.1", src_port=4444,
               dst_port=80, protocol=6, packet_size=520, timestamp=2.0),
    NO_DEADLINE))
check("H1: refused kernel DROP committed no mirror layer",
      ATTACKER_H not in engine_h.get_ephemeral_blacklist()
      and ATTACKER_H not in engine_h.get_blacklist()
      and ATTACKER_H not in inter_h._blocked,
      f"ephemeral={engine_h.get_ephemeral_blacklist()} "
      f"blacklist={engine_h.get_blacklist()} blocked={inter_h._blocked}")
check("H2: the ban is queued for retry instead of being forgotten",
      ATTACKER_H in inter_h._pending_enforce,
      f"pending_enforce={sorted(inter_h._pending_enforce)}")

stub_h.refusing = False
asyncio.run(inter_h._retry_pending())
check("H3: the sweeper retry installs the DROP and only then the mirror",
      ATTACKER_H in stub_h.blocked
      and ATTACKER_H in engine_h.get_ephemeral_blacklist()
      and ATTACKER_H in inter_h._blocked
      and ATTACKER_H not in inter_h._pending_enforce,
      f"blocked={stub_h.blocked} "
      f"ephemeral={engine_h.get_ephemeral_blacklist()} "
      f"pending_enforce={sorted(inter_h._pending_enforce)}")

# ---------------------------------------------------------------- Bug C
print()
print("=" * 60)
print("Bug C: kernel redirect queue must match the userspace listener")
print("=" * 60)
import shutil

if shutil.which("iptables") is None:
    # setup() rightfully refuses to start without the iptables binary —
    # nothing kernel-side to validate on this host (e.g. macOS dev machine).
    print("[SKIP] C1/C2: iptables not in PATH — kernel redirect checks "
          "require a Linux host")
else:
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
