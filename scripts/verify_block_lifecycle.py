#!/usr/bin/env python3
"""Cross-validation: ban-lifecycle bugs found in the E2E health pass.

Bug D1  Rule-engine BLOCK verdicts (blacklist hit / rate limit / protocol)
        fed the strike-escalation policy.  An operator-blacklisted IP
        accumulated strikes until a temp ban "expired" and wiped the
        operator's entry from the blacklist (persisted on engine stop).
        Fix: _handle() only counts strikes for ML-detector BLOCKs.

Bug D2  Rate-limit false positives escalated the same way: a legitimate
        >100/s source was temp-banned, then perm-banned into rules.json.
        Fix: same gate as D1 — rule-engine verdicts never escalate.

Bug D3  Temp-ban expiry removed the rule-engine blacklist entry
        unconditionally, so a mirror installed by the ban erased any
        pre-existing (operator) entry for the same IP.
        Fix: temp bans mirror only when the entry is new, track mirror
        provenance in Interceptor._ban_mirrors, and the sweeper removes
        only mirrored entries.  An operator blacklisting a source mid-ban
        (API POST /rules/blacklist) promotes the entry via
        note_operator_blacklist() so the sweeper leaves it alone.

Bug D4  `cli.py block/unblock/whitelist` edited only the CLI process's
        local pipeline + rules.json; a running engine never saw the change,
        and `unblock` left the kernel DROP in place.
        Fix: the three commands go through the management API and fall back
        to local editing with an explicit warning when it is unreachable.

Bug E   One IP failing to unblock killed the temp-ban sweeper task (the
        BlockPolicy had already forgotten the ban, so every later expiry
        was stranded in kernel + blacklist with nothing left to lift it).
        Fix: _lift_temp_ban() clears each enforcement layer best-effort;
        a single IP's failure can no longer kill the loop.

Bug F   loopback / safe-ips sources were refused a kernel DROP by
        IptablesManager, but the escalation paths still mirrored them into
        the blacklist and wrote them to rules.json — a persistent phantom
        block the kernel never enforced.
        Fix: blockable()/is_blockable() apply one network-aware test
        everywhere; the interceptor skips kernel+mirror+persistence for
        un-blockable sources (the packet is still dropped inline), and
        app.py rejects such entries at the API boundary + sweeps them at
        startup.

Bug G   A detector raising in process_packet() took down the whole chain:
        the interceptor's catch-all dropped EVERY packet (self-DoS).
        Fix: per-detector fault isolation — a raising detector abstains
        for that packet; after 5 consecutive exceptions the circuit
        breaker skips it until restart (visible in status()).

Exit 0 = all fixes verified; exit 1 = at least one check failed.
"""
import contextlib
import io
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import networksecurity.interception.interceptor as interceptor_mod
from networksecurity.engine.block_policy import BlockPolicy
from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action
from networksecurity.interception.interceptor import Interceptor
from networksecurity.interception.iptables import blockable

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    if not cond:
        ok = False
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


class Blocker(BaseDetector):
    """Always-BLOCK stand-in for an ML detector."""

    async def process_packet(self, packet: PacketInfo):
        from networksecurity.engine.verdict import Action, ThreatLevel, Verdict
        return Verdict(action=Action.BLOCK, confidence=0.99,
                       threat_level=ThreatLevel.HIGH,
                       reason="anomaly", detector=self.name)


class StubIptables:
    """Records block/unblock calls; no kernel involvement.

    Applies the real blockable() criteria so tests exercise the same
    loopback/safe-ips refusals the kernel manager applies.
    """

    def __init__(self, safe_ips=None):
        self.blocked: list[str] = []
        self._safe_ips = safe_ips or []

    def block_ip(self, ip: str) -> None:
        if not blockable(ip, self._safe_ips):
            return
        if ip not in self.blocked:
            self.blocked.append(ip)

    def unblock_ip(self, ip: str) -> None:
        with contextlib.suppress(ValueError):
            self.blocked.remove(ip)

    def is_blockable(self, ip: str) -> bool:
        return blockable(ip, self._safe_ips)

    def cleanup_all(self) -> None:
        self.blocked.clear()


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt: float):
        self.t += dt


def pkt(src: str, t: float) -> PacketInfo:
    return PacketInfo(src_ip=src, dst_ip="10.0.0.1", src_port=4444,
                      dst_port=80, protocol=6, packet_size=520,
                      timestamp=t)


def make_interceptor(rule_engine: RuleEngine, policy: BlockPolicy,
                     ml: bool = True) -> Interceptor:
    pipeline = DetectionPipeline(rule_engine=rule_engine)
    if ml:
        pipeline.add_detector(Blocker(name="FakeML"))
    inter = Interceptor(pipeline, block_policy=policy)
    inter._iptables = StubIptables()  # type: ignore[assignment]
    return inter


async def send(inter: Interceptor, packets: list[PacketInfo]) -> None:
    for p in packets:
        await inter._handle(p, {"timed_out": False})


async def run_sweeper_once(inter: Interceptor) -> None:
    """Run _temp_ban_sweeper() long enough to process one expiry cycle."""
    real_asyncio = interceptor_mod.asyncio

    async def fast_sleep(delay, *a, **k):
        if delay >= 30.0:            # the sweeper's periodic wait: yield, don't sleep
            await real_asyncio.sleep(0)
            return
        await real_asyncio.sleep(delay, *a, **k)

    interceptor_mod.asyncio = types.SimpleNamespace(sleep=fast_sleep)
    task = real_asyncio.create_task(inter._temp_ban_sweeper())
    try:
        for _ in range(5):
            await real_asyncio.sleep(0)
    finally:
        task.cancel()
        interceptor_mod.asyncio = real_asyncio
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------- D1 + D2
print("=" * 60)
print("D1/D2: rule-engine BLOCKs never feed the escalation policy")
print("=" * 60)
clock = Clock()
policy = BlockPolicy(strikes_threshold=5, strikes_window=300.0,
                     temp_ban_seconds=600.0, temp_ban_count_to_perm=3,
                     now=clock)

ATTACKER = "203.0.113.66"
engine = RuleEngine()
engine.add_blacklist(ATTACKER)          # operator's persistent entry
inter = make_interceptor(engine, policy)
asyncio.run(send(inter, [pkt(ATTACKER, clock() + i * 0.01) for i in range(10)]))

check("D1a: 10 blacklist-hit BLOCKs produced no strike record",
      policy.get(ATTACKER) is None,
      f"record={policy.get(ATTACKER)}")
check("D1b: operator blacklist entry untouched",
      engine.get_blacklist() == [ATTACKER],
      f"blacklist={engine.get_blacklist()}")
check("D1c: no kernel block installed",
      ATTACKER not in inter._iptables.blocked)

# Rate limit: cap 1/s.  No ML detector in the chain — the first packet is
# allowed by the limiter, the other 6 are rate-limit BLOCKs; all of those
# must be gated out of the strike policy.
rl_engine = RuleEngine(window_seconds=1.0, max_connections=1)
rl_inter = make_interceptor(rl_engine, policy, ml=False)
BUSY = "198.51.100.7"
asyncio.run(send(rl_inter, [pkt(BUSY, 2_000.0 + i * 0.01) for i in range(7)]))
check("D2a: 6 rate-limit BLOCKs produced no strike record",
      policy.get(BUSY) is None,
      f"record={policy.get(BUSY)}")
check("D2b: rate-limited source NOT blacklisted",
      BUSY not in rl_engine.get_blacklist(),
      f"blacklist={rl_engine.get_blacklist()}")
check("D2c: no kernel block for the rate-limited source",
      BUSY not in rl_inter._iptables.blocked)

# ---------------------------------------------------------------- D3
print()
print("=" * 60)
print("D3: temp-ban mirror lifecycle is fully reversible")
print("=" * 60)
clock3 = Clock()
policy3 = BlockPolicy(strikes_threshold=5, strikes_window=300.0,
                      temp_ban_seconds=600.0, temp_ban_count_to_perm=3,
                      now=clock3)
engine3 = RuleEngine()
inter3 = make_interceptor(engine3, policy3)
VICTIM = "198.51.100.10"
asyncio.run(send(inter3, [pkt(VICTIM, clock3() + i * 0.01) for i in range(5)]))
check("D3a: 5 ML BLOCKs escalated to a temp ban",
      (r := policy3.get(VICTIM)) is not None and r.state == "temp_banned",
      f"record={policy3.get(VICTIM)}")
check("D3b: temp ban mirrored into the rule-engine blacklist",
      VICTIM in engine3.get_blacklist())
check("D3c: kernel DROP installed",
      VICTIM in inter3._iptables.blocked)

clock3.advance(601.0)
asyncio.run(run_sweeper_once(inter3))
check("D3d: sweeper removed the mirror from the blacklist",
      VICTIM not in engine3.get_blacklist(),
      f"blacklist={engine3.get_blacklist()}")
check("D3e: sweeper lifted the kernel DROP",
      VICTIM not in inter3._iptables.blocked)
check("D3f: mirror provenance cleaned up",
      VICTIM not in inter3._ban_mirrors)

# ---------------------------------------------------------------- D3 (promote)
print()
print("=" * 60)
print("D3b: an operator entry blacklisted MID-ban survives expiry")
print("=" * 60)
clock4 = Clock()
policy4 = BlockPolicy(strikes_threshold=5, strikes_window=300.0,
                      temp_ban_seconds=600.0, temp_ban_count_to_perm=3,
                      now=clock4)
engine4 = RuleEngine()
inter4 = make_interceptor(engine4, policy4)
MIDBAN = "198.51.100.20"
asyncio.run(send(inter4, [pkt(MIDBAN, clock4() + i * 0.01) for i in range(5)]))
# Operator blacklists the source while the temp ban is active (API POST).
engine4.add_blacklist(MIDBAN)
inter4.note_operator_blacklist(MIDBAN)

clock4.advance(601.0)
asyncio.run(run_sweeper_once(inter4))
check("D3g: operator entry survives the temp-ban expiry",
      MIDBAN in engine4.get_blacklist(),
      f"blacklist={engine4.get_blacklist()}")
check("D3h: kernel DROP still lifted at expiry",
      MIDBAN not in inter4._iptables.blocked)

# ---------------------------------------------------------------- D5
print()
print("=" * 60)
print("D5: ML escalation to a permanent ban still works end to end")
print("=" * 60)
tmp_rules = Path(__file__).resolve().parent.parent / "scripts" / ".verify_tmp_rules.json"
tmp_rules.unlink(missing_ok=True)
real_rules_file = interceptor_mod.RULES_FILE
interceptor_mod.RULES_FILE = tmp_rules
try:
    clock5 = Clock()
    policy5 = BlockPolicy(strikes_threshold=5, strikes_window=300.0,
                          temp_ban_seconds=600.0, temp_ban_count_to_perm=1,
                          now=clock5)
    engine5 = RuleEngine()
    inter5 = make_interceptor(engine5, policy5)
    BAD = "198.51.100.30"
    # Ban 1: 5 ML BLOCKs -> temp ban (mirror takes over userspace enforcement).
    asyncio.run(send(inter5, [pkt(BAD, clock5() + i * 0.01) for i in range(5)]))
    # The ban expires; the repeat offender must now re-earn the threshold.
    clock5.advance(601.0)
    asyncio.run(run_sweeper_once(inter5))
    asyncio.run(send(inter5, [pkt(BAD, clock5() + i * 0.01) for i in range(5)]))
    rec5 = policy5.get(BAD)
    check("D5a: repeat offending escalated straight to perm_banned",
          rec5 is not None and rec5.state == "perm_banned",
          f"record={rec5}")
    check("D5b: perm ban present in the rule-engine blacklist",
          BAD in engine5.get_blacklist())
    check("D5c: perm ban persisted into rules.json",
          tmp_rules.exists() and BAD in json.loads(tmp_rules.read_text()).get("blacklist", []),
          f"rules.json={tmp_rules.read_text().strip() if tmp_rules.exists() else '(missing)'}")
    check("D5d: kernel DROP installed for the perm ban",
          BAD in inter5._iptables.blocked)
finally:
    interceptor_mod.RULES_FILE = real_rules_file
    tmp_rules.unlink(missing_ok=True)

# ---------------------------------------------------------------- D4
print()
print("=" * 60)
print("D4: cli block/unblock/whitelist fall back loudly when the API is down")
print("=" * 60)
import cli as cli_mod

tmp_cli_rules = Path(__file__).resolve().parent.parent / "scripts" / ".verify_tmp_cli_rules.json"
tmp_cli_rules.unlink(missing_ok=True)
real_cli_rules = cli_mod.RULES_FILE
real_api_request = cli_mod._api_request
cli_mod.RULES_FILE = tmp_cli_rules


def _api_down(*a, **k):
    raise ConnectionError("Connection refused")


cli_mod._api_request = _api_down
try:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        cli_mod.cmd_block(types.SimpleNamespace(ip="203.0.113.66"))
    check("D4a: block fell back to local rules.json",
          "203.0.113.66" in cli_mod.pipeline.rule_engine.get_blacklist())

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        cli_mod.cmd_whitelist(types.SimpleNamespace(ip="10.9.9.0/24"))
        cli_mod.cmd_unblock(types.SimpleNamespace(ip="203.0.113.66"))
    out_s, err_s = out.getvalue(), err.getvalue()

    check("D4b: whitelist fell back to local rules.json",
          "10.9.9.0/24" in cli_mod.pipeline.rule_engine.get_whitelist())
    check("D4c: unblock fell back to local rules.json",
          "203.0.113.66" not in cli_mod.pipeline.rule_engine.get_blacklist())
    check("D4d: fallback announced a WARNING on stderr",
          err_s.count("WARNING") >= 3, f"stderr={err_s.strip()!r}")
    check("D4e: fallback wrote the local file",
          tmp_cli_rules.exists()
          and "10.9.9.0/24" in json.loads(tmp_cli_rules.read_text()).get("whitelist", []))
finally:
    cli_mod.RULES_FILE = real_cli_rules
    cli_mod._api_request = real_api_request
    tmp_cli_rules.unlink(missing_ok=True)

# ---------------------------------------------------------------- E
print()
print("=" * 60)
print("E: one IP's expiry failure does not strand the sweeper or other IPs")
print("=" * 60)


class FlakyUnblockStub(StubIptables):
    """Raises once on fail_ip's unblock, like a transient iptables error."""

    def __init__(self, fail_ip: str):
        super().__init__()
        self.fail_ip = fail_ip
        self.failed = False

    def unblock_ip(self, ip: str) -> None:
        if ip == self.fail_ip and not self.failed:
            self.failed = True
            raise RuntimeError("simulated iptables failure")
        super().unblock_ip(ip)


clockE = Clock()
policyE = BlockPolicy(strikes_threshold=1, temp_ban_seconds=600.0,
                      temp_ban_count_to_perm=99, now=clockE)
engineE = RuleEngine()
interE = make_interceptor(engineE, policyE)
E1, E2 = "198.51.100.40", "198.51.100.41"
interE._iptables = FlakyUnblockStub(E1)  # type: ignore[assignment]
asyncio.run(send(interE, [pkt(E1, clockE() + 0.01), pkt(E2, clockE() + 0.02)]))
check("E0a: both sources temp-banned",
      (rE1 := policyE.get(E1)) is not None and rE1.state == "temp_banned"
      and (rE2 := policyE.get(E2)) is not None and rE2.state == "temp_banned",
      f"records={policyE.get(E1)}, {policyE.get(E2)}")
check("E0b: both mirrors installed",
      E1 in engineE.get_blacklist() and E2 in engineE.get_blacklist(),
      f"blacklist={engineE.get_blacklist()}")

clockE.advance(601.0)
asyncio.run(run_sweeper_once(interE))
# Reaching these checks at all proves the sweeper task survived: a raise
# inside _temp_ban_sweeper propagates through run_sweeper_once's await.
check("E1a: healthy IP's kernel DROP lifted in the same cycle",
      E2 not in interE._iptables.blocked,
      f"blocked={interE._iptables.blocked}")
check("E1b: healthy IP's blacklist mirror lifted",
      E2 not in engineE.get_blacklist())
check("E1c: healthy IP's bookkeeping cleared",
      E2 not in interE._blocked and E2 not in interE._ban_mirrors)
check("E1d: failed IP's kernel DROP left visible for reconciliation",
      E1 in interE._iptables.blocked)
check("E1e: failed IP's mirror and bookkeeping still cleared best-effort",
      E1 not in engineE.get_blacklist()
      and E1 not in interE._blocked and E1 not in interE._ban_mirrors)

# ---------------------------------------------------------------- F
print()
print("=" * 60)
print("F: loopback / safe-ips sources never reach kernel, mirror, rules.json")
print("=" * 60)

# F1: loopback on the temp-ban path (threshold=1 -> first ML BLOCK enforces).
clockF = Clock()
policyF = BlockPolicy(strikes_threshold=1, temp_ban_seconds=600.0,
                      temp_ban_count_to_perm=99, now=clockF)
engineF = RuleEngine()
interF = make_interceptor(engineF, policyF)
LOOP = "127.0.0.53"
dropped = asyncio.run(interF._handle(pkt(LOOP, clockF() + 0.01),
                                     {"timed_out": False}))
check("F1a: loopback packet still dropped inline", dropped is True)
check("F1b: no kernel DROP for the loopback source",
      LOOP not in interF._iptables.blocked)
check("F1c: no blacklist mirror for the loopback source",
      LOOP not in engineF.get_blacklist())
check("F1d: no _blocked/_ban_mirrors bookkeeping",
      LOOP not in interF._blocked and LOOP not in interF._ban_mirrors)

# F2: loopback escalating all the way to a perm ban must not touch rules.json.
tmp_rules_f = Path(__file__).resolve().parent.parent / "scripts" / ".verify_tmp_rules_f.json"
tmp_rules_f.unlink(missing_ok=True)
real_rules_f = interceptor_mod.RULES_FILE
interceptor_mod.RULES_FILE = tmp_rules_f
try:
    policyF2 = BlockPolicy(strikes_threshold=1, temp_ban_seconds=600.0,
                           temp_ban_count_to_perm=1, now=clockF)
    engineF2 = RuleEngine()
    interF2 = make_interceptor(engineF2, policyF2)
    # Ban 1: temp (guarded away).  After expiry, re-offending with
    # temp_ban_count_to_perm=1 escalates straight to perm.
    asyncio.run(send(interF2, [pkt(LOOP, clockF() + 1.01)]))
    clockF.advance(601.0)
    asyncio.run(run_sweeper_once(interF2))
    asyncio.run(send(interF2, [pkt(LOOP, clockF() + 1.02)]))
    recF = policyF2.get(LOOP)
    check("F2a: policy still escalated to perm_banned",
          recF is not None and recF.state == "perm_banned", f"record={recF}")
    check("F2b: perm ban wrote nothing to rules.json",
          not tmp_rules_f.exists()
          or LOOP not in json.loads(tmp_rules_f.read_text()).get("blacklist", []),
          f"rules.json={tmp_rules_f.read_text().strip() if tmp_rules_f.exists() else '(missing)'}")
    check("F2c: no kernel DROP / _blocked entry for the perm ban",
          LOOP not in interF2._iptables.blocked and LOOP not in interF2._blocked)
finally:
    interceptor_mod.RULES_FILE = real_rules_f
    tmp_rules_f.unlink(missing_ok=True)

# F3: an interception.safe_ips entry is honored on the escalation paths.
SAFE = "198.51.100.99"
engineF3 = RuleEngine()
interF3 = make_interceptor(engineF3, policyF)
interF3._iptables = StubIptables(safe_ips=[SAFE])  # type: ignore[assignment]
asyncio.run(send(interF3, [pkt(SAFE, clockF() + 2.01)]))
check("F3a: no kernel DROP for the safe-ips source",
      SAFE not in interF3._iptables.blocked)
check("F3b: no blacklist mirror for the safe-ips source",
      SAFE not in engineF3.get_blacklist())

# ---------------------------------------------------------------- G
print()
print("=" * 60)
print("G: a poisoned detector is isolated, not fatal to the chain")
print("=" * 60)


class Poison(BaseDetector):
    """Raises for the first ``fails`` calls (None = forever), then abstains."""

    def __init__(self, name: str, fails: int | None = None):
        super().__init__(name=name)
        self.calls = 0
        self._fails = fails

    async def process_packet(self, packet: PacketInfo):
        self.calls += 1
        if self._fails is None or self.calls <= self._fails:
            raise RuntimeError("poisoned detector")
        return None


pipe_g = DetectionPipeline(rule_engine=RuleEngine())
poison_always = Poison(name="Poison", fails=None)
poison_flaky = Poison(name="Flaky", fails=4)
pipe_g.add_detector(poison_always)
pipe_g.add_detector(poison_flaky)
pipe_g.add_detector(Blocker(name="Healthy"))

verdicts = asyncio.run(
    pipe_g.process_batch([pkt("192.0.2.55", 5_000.0 + i * 0.01) for i in range(6)])
)
check("G1: chain still decides BLOCK via the healthy detector on every packet",
      all(v.action == Action.BLOCK and v.detector == "Healthy" for v in verdicts),
      f"verdicts={[(v.detector, v.action.value) for v in verdicts]}")
status_g = pipe_g.status()
check("G2: always-raising detector tripped the breaker (5 consecutive failures)",
      status_g["broken_detectors"] == ["Poison"],
      f"broken={status_g['broken_detectors']}")
check("G3: recovering detector never tripped (counter resets on success)",
      "Flaky" not in status_g["broken_detectors"])
check("G4: tripped detector is no longer invoked",
      poison_always.calls == DetectionPipeline.FAILURE_THRESHOLD,
      f"calls={poison_always.calls}")
check("G5: recovering detector kept receiving packets",
      poison_flaky.calls == 6, f"calls={poison_flaky.calls}")

print()
print("=" * 60)
print("RESULT:", "ALL FIXES VERIFIED" if ok else "FAILURES PRESENT (see FAIL lines)")
print("=" * 60)
sys.exit(0 if ok else 1)
