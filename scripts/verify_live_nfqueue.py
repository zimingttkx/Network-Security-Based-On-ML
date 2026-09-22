#!/usr/bin/env python3
"""Live end-to-end checks: real packets, real kernel, real verdicts.

Everything else in this directory stops at our own boundary.  The unit tests
prove that ``iptables -I`` was called with the right arguments, and
verify_kernel_rules.py proves the kernel ends up holding those rules.  Neither
proves the thing that matters at 3 a.m.: that a frame arriving on an interface
is handed to userspace, run through the pipeline, and answered with a verdict
the kernel then enforces — or that stopping leaves the host reachable again.

So this runs the real Interceptor against traffic that crosses a link:

    nips-cli  veth-cli <==veth==> veth-srv  nips-srv
    10.77.0.2                               10.77.0.1
    fd00:77::2                              fd00:77::1

It must run inside the *server* namespace (the CI job builds the topology):

    sudo ip netns exec nips-srv python scripts/verify_live_nfqueue.py

Client-side probes are subprocesses in the other namespace, so nothing is
injected or looped: a connection that completes while the NFQUEUE redirect is
installed can only have completed because a userspace consumer drained the
queue, and a redirect left behind by teardown shows up as that same connection
failing.

One piece is a fixture rather than the product: the ban ladder is driven by
``EscalationTrigger``, a detector that blocks on a dedicated port.  See its
docstring for why a rule-engine verdict cannot drive that path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# Works under `sudo ip netns exec ... python scripts/...`, where PYTHONPATH is
# reset and sys.path[0] is scripts/ rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.engine.detector import BaseDetector, PacketInfo  # noqa: E402
from networksecurity.engine.verdict import (  # noqa: E402
    Action,
    ThreatLevel,
    Verdict,
)

SERVER_V4 = os.environ.get("NIPS_LIVE_SERVER_V4", "10.77.0.1")
CLIENT_V4 = os.environ.get("NIPS_LIVE_CLIENT_V4", "10.77.0.2")
SERVER_V6 = os.environ.get("NIPS_LIVE_SERVER_V6", "fd00:77::1")
CLIENT_V6 = os.environ.get("NIPS_LIVE_CLIENT_V6", "fd00:77::2")
SERVER_NS = os.environ.get("NIPS_LIVE_SERVER_NS", "nips-srv")
CLIENT_NS = os.environ.get("NIPS_LIVE_NS", "nips-cli")
SERVER_DEV = os.environ.get("NIPS_LIVE_SERVER_DEV", "veth-srv")
CLIENT_DEV = os.environ.get("NIPS_LIVE_CLIENT_DEV", "veth-cli")

TCP_PORT = 8099          # ordinary service port
GUARDED_PORT = 22        # the ssh guard rule matches exactly this dport
V6_PORT = 8100
QUEUE_A = 41
QUEUE_B = 42
# Port the escalation trigger blocks on.  Dedicated so the ban ladder can be
# driven without disturbing the probes that must keep reaching the listeners.
ESCALATION_PORT = 8098

# The rate limiter counts new connections only (TCP SYN, UDP datagrams), so a
# flood is a number of connection attempts rather than of packets.
RATE_MAX = 120
RATE_WINDOW = 10.0
# Deliberately below RATE_MAX: the flood has to reach the escalation trigger
# rather than be short-circuited by the rule engine's rate limit, which is
# enforced inline and never escalates.
TRIGGER_FLOOD = 20
TEMP_BAN_SECONDS = 4.0
# The expiry sweeper runs on a fixed 30 s cycle and only *then* removes the
# DROP, so a lifted ban cannot be observed before the next tick.
LIFT_WAIT = 90.0
# Once the ban is installed, in-flight packets are what carries the policy
# past its own TTL; give the kernel a moment before trusting the absence.
BAN_WAIT = 20.0

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL':10} {name}" + (f"  [{detail}]" if detail else ""))


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def in_client(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    return subprocess.run(("ip", "netns", "exec", CLIENT_NS, *args),
                          capture_output=True, text=True, timeout=timeout)


def rules(tool: str) -> str:
    return sh(tool, "-S")


def drop_present(tool: str, ip: str) -> bool:
    """True when a DROP rule for this source is installed (in either family)."""
    needle = ip.lower()
    return any(needle in line and line.rstrip().endswith("-j DROP")
               for line in rules(tool).splitlines())


def wait_for(fn, timeout: float, interval: float = 1.0) -> float:
    """Poll ``fn`` until it is true; return how long that took, or -1.0."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if fn():
            return time.monotonic() - started
        time.sleep(interval)
    return -1.0


# --- client-side probes ------------------------------------------------------

_CONNECT = """
import socket, sys
host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
fam = socket.AF_INET6 if ':' in host else socket.AF_INET
s = socket.socket(fam, socket.SOCK_STREAM)
s.settimeout(timeout)
try:
    s.connect((host, port))
    s.sendall(b'probe')
    if not s.recv(16):
        raise OSError('empty reply')
except OSError as exc:
    print(type(exc).__name__)
    sys.exit(1)
print('ok')
"""

_FLOOD = """
import socket, sys
host, port, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fam = socket.AF_INET6 if ':' in host else socket.AF_INET
s = socket.socket(fam, socket.SOCK_DGRAM)
for _ in range(count):
    s.sendto(b'x' * 40, (host, port))
print(count)
"""


def tcp_probe(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Open a real TCP connection to ``host`` from the client namespace."""
    rc = in_client(sys.executable, "-c", _CONNECT, host, str(port), str(timeout),
                   timeout=timeout + 15)
    out = (rc.stdout or rc.stderr).strip()
    return rc.returncode == 0, out.splitlines()[-1] if out else "no output"


def flood(host: str, port: int, count: int) -> None:
    in_client(sys.executable, "-c", _FLOOD, host, str(port), str(count))


# --- server-side listeners ---------------------------------------------------

class Listeners:
    """Echo sockets the client probes connect to; counts accepted sessions."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.accepted: dict[int, int] = {}
        self._lock = threading.Lock()

    def add(self, port: int, family: int = socket.AF_INET) -> None:
        srv = socket.socket(family, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0" if family == socket.AF_INET else "::", port))
        srv.listen(32)
        srv.settimeout(0.4)
        thread = threading.Thread(target=self._serve, args=(srv, port), daemon=True)
        thread.start()
        self._threads.append(thread)

    def _serve(self, srv: socket.socket, port: int) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with self._lock:
                self.accepted[port] = self.accepted.get(port, 0) + 1
            with conn:
                try:
                    conn.settimeout(2.0)
                    if conn.recv(16):
                        conn.sendall(b"pong")
                except OSError:
                    pass
        srv.close()

    def count(self, port: int) -> int:
        with self._lock:
            return self.accepted.get(port, 0)

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3.0)


# --- interceptor under test --------------------------------------------------

class EscalationTrigger(BaseDetector):
    """A detector that blocks on demand, so the ban ladder can be driven.

    The ladder (strike -> temp ban -> expiry -> perm ban) only ever runs for
    verdicts from outside the rule engine: the interceptor refuses to escalate
    a rule-engine BLOCK, because those are already enforced inline on every
    packet and escalating them let a rate limit or an operator's own blacklist
    entry turn into a permanent, persisted ban.  So to watch a real kernel DROP
    appear and be lifted on a real host, something has to *decide* — this does,
    on a dedicated port, and nothing else.

    Everything downstream of the verdict is product code: the strike policy,
    the iptables DROP, the blacklist mirror, rules.json and the expiry sweeper.
    """

    def __init__(self, port: int) -> None:
        super().__init__(name="escalation-trigger")
        self._port = port

    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        if packet.protocol == 17 and packet.dst_port == self._port:
            return Verdict(action=Action.BLOCK, confidence=0.99,
                           threat_level=ThreatLevel.HIGH,
                           reason="escalation trigger", detector=self.name)
        return None


def build_interceptor(rules_file: Path, queue_num: int, intercept_icmp: bool,
                      verdicts: list[tuple[str, str, str]]):
    from networksecurity.engine import RuleEngine
    from networksecurity.engine.block_policy import BlockPolicy
    from networksecurity.engine.pipeline import DetectionPipeline
    from networksecurity.interception import Interceptor
    import networksecurity.interception.interceptor as mod

    # A permanent ban persists to this path; at its default the suite would
    # rewrite the developer's own rules.json the first time it escalates.
    mod.RULES_FILE = rules_file

    pipeline = DetectionPipeline()
    # The rule engine is the only detector that decides on real traffic: the
    # escalation trigger abstains on everything except its own port, so a probe
    # that completes is one the rules let through rather than a fail-closed
    # accident.
    pipeline.set_rule_engine(RuleEngine(
        window_seconds=RATE_WINDOW,
        max_connections=RATE_MAX,
        allowed_protocols={6, 17},
        allowed_icmp_types={8},
    ))
    pipeline.add_detector(EscalationTrigger(ESCALATION_PORT))
    # One strike is enough to ban, and two completed bans make it permanent, so
    # the temp-ban -> lift -> ban-again -> persist ladder needs two floods and
    # the blow that starts the third round.
    policy = BlockPolicy(strikes_threshold=1, strikes_window=600.0,
                         temp_ban_seconds=TEMP_BAN_SECONDS,
                         temp_ban_count_to_perm=2, table_max=100)

    def _record(packet, verdict) -> None:
        verdicts.append((packet.src_ip, verdict.action.value, verdict.reason))

    return Interceptor(pipeline, queue_num=queue_num, safe_ips=["10.78.0.0/24"],
                       on_verdict=_record, block_policy=policy,
                       intercept_icmp=intercept_icmp)


def start(inter, queue_num: int) -> None:
    """setup() + capture on a background thread, the split app.py uses."""
    inter.setup()
    threading.Thread(target=inter.begin_capture, daemon=True).start()
    # The redirect has to be live before the first probe, otherwise a passing
    # connection proves nothing about the queue.
    waited = wait_for(lambda: f"-j NFQUEUE --queue-num {queue_num}"
                      in rules("iptables"), 10.0, 0.2)
    check("the NFQUEUE redirect is installed before probing", waited >= 0,
          f"after {waited:.1f}s")


def stop(inter) -> None:
    inter.stop()
    inter._nfqueue._stopped.wait(5.0)


def is_blocked(verdicts: list, ip: str, reason: str) -> bool:
    return any(src == ip and action == "block" and reason in text.lower()
               for src, action, text in verdicts)


def persisted(rules_file: Path) -> list[str]:
    try:
        return json.loads(rules_file.read_text()).get("blacklist", [])
    except (OSError, ValueError):
        return []


# --- phases ------------------------------------------------------------------

def phase_baseline(chain: str) -> None:
    print("\n# baseline: the topology works with no rules installed")
    check("our chain is absent before the run", chain not in rules("iptables"))
    ok, detail = tcp_probe(SERVER_V4, TCP_PORT)
    check("v4 TCP reaches the listener", ok, detail)
    ok, detail = tcp_probe(SERVER_V6, V6_PORT)
    check("v6 TCP reaches the listener", ok, detail)


def phase_consumption(inter, chain: str, listeners: Listeners,
                      verdicts: list) -> None:
    print("\n# userspace really consumes what the kernel queues")
    st = inter.status()
    jumps = rules("iptables").count(f"-A INPUT -j {chain}")
    check("INPUT jumps to the NIPS chain exactly once", jumps == 1, str(jumps))
    check("IPv6 is reported ready", st["ipv6_ready"] is True)

    before = inter.status()["nfqueue_packets"]
    ok, detail = tcp_probe(SERVER_V4, TCP_PORT)
    check("v4 connects while every SYN goes through NFQUEUE", ok, detail)
    queued = inter.status()["nfqueue_packets"] - before
    check("the connection was consumed by userspace, not bypassed", queued > 0,
          f"{queued} packets queued")
    check("every queued packet parsed",
          inter.status()["nfqueue_parse_failed"] == 0)
    check("the detection loop answered",
          inter.status()["detection_loop_stale_seconds"] is not None)
    check("nothing was dropped on the way", inter.status()["nfqueue_dropped"] == 0)
    check("the verdict callback saw the client",
          any(src == CLIENT_V4 for src, _, _ in verdicts))

    # The guard rules sit ahead of the redirect, so traffic on a guarded port
    # must reach the application without ever entering the queue.
    before = inter.status()["nfqueue_packets"]
    accepted = listeners.count(GUARDED_PORT)
    ok, detail = tcp_probe(SERVER_V4, GUARDED_PORT)
    check("the dport-22 guard lets ssh-class traffic through", ok, detail)
    after = inter.status()["nfqueue_packets"]
    check("and it never enters the queue",
          after == before and listeners.count(GUARDED_PORT) == accepted + 1,
          f"queued delta={after - before}")

    before = inter.status()["nfqueue_packets"]
    rc = in_client("ping", "-c", "1", "-W", "2", SERVER_V4)
    last = (rc.stdout or "").strip().splitlines()
    check("ICMP still answers with interception on", rc.returncode == 0,
          last[-1] if last else f"rc={rc.returncode}")
    check("and is not queued while intercept_icmp is off",
          inter.status()["nfqueue_packets"] == before)


def phase_enforcement(inter, verdicts: list, rules_file: Path) -> None:
    print("\n# the block ladder runs on the real kernel")
    flood(SERVER_V4, ESCALATION_PORT, TRIGGER_FLOOD)
    took = wait_for(lambda: drop_present("iptables", CLIENT_V4), BAN_WAIT, 0.5)
    check("a BLOCK verdict installed a kernel DROP", took >= 0,
          f"after {took:.1f}s")
    check("the DROP is attributed to the detector that blocked",
          is_blocked(verdicts, CLIENT_V4, "escalation trigger"))
    check("blocked_ips reports the source",
          CLIENT_V4 in inter.status()["blocked_ips"],
          str(inter.status()["blocked_ips"]))
    check("the kernel refused nothing", inter.status()["pending_enforce"] == [])
    ok, detail = tcp_probe(SERVER_V4, TCP_PORT, timeout=3.0)
    check("traffic from the banned source really stops arriving", not ok, detail)

    took = wait_for(lambda: not drop_present("iptables", CLIENT_V4), LIFT_WAIT, 2.0)
    check("a temp ban lifts itself when it expires", took >= 0, f"after {took:.1f}s")
    ok, detail = tcp_probe(SERVER_V4, TCP_PORT)
    check("and the source can connect again", ok, detail)

    # A repeat offender is escalated to permanent and written to rules.json.
    # Every ban cycle has to be re-earned: expiry clears the strikes but keeps
    # the completed-ban counter, so this takes three rounds — the ban that
    # opens the cycle, the one that spends the counter's last unit, and the
    # strike after it expires that finds nothing left to rotate through.
    escalated = False
    for _ in range(3):
        flood(SERVER_V4, ESCALATION_PORT, TRIGGER_FLOOD)
        if wait_for(lambda: drop_present("iptables", CLIENT_V4), BAN_WAIT, 0.5) < 0:
            break
        wait_for(lambda: CLIENT_V4 in persisted(rules_file)
                 or not drop_present("iptables", CLIENT_V4), LIFT_WAIT, 2.0)
        if CLIENT_V4 in persisted(rules_file):
            escalated = True
            break
    check("a repeat offender ends up permanently banned", escalated)
    check("the permanent ban is what the rules file persists",
          CLIENT_V4 in persisted(rules_file), str(persisted(rules_file)))
    ok, detail = tcp_probe(SERVER_V4, TCP_PORT, timeout=3.0)
    check("and it stays blocked", not ok, detail)


def phase_ipv6(inter, chain: str, verdicts: list) -> None:
    print("\n# IPv6 is on the same footing")
    check("the v6 chain exists", chain in rules("ip6tables"), chain)
    check("the v6 redirect is installed",
          f"-A {chain} -p tcp -j NFQUEUE" in rules("ip6tables"))

    before = inter.status()["nfqueue_packets"]
    ok, detail = tcp_probe(SERVER_V6, V6_PORT)
    check("v6 connects while its packets go through NFQUEUE", ok, detail)
    check("v6 traffic was consumed, not bypassed",
          inter.status()["nfqueue_packets"] > before)
    check("the v6 source reached the pipeline",
          any(src == CLIENT_V6 for src, _, _ in verdicts),
          str(sorted({s for s, _, _ in verdicts})))

    flood(SERVER_V6, ESCALATION_PORT, TRIGGER_FLOOD)
    took = wait_for(lambda: drop_present("ip6tables", CLIENT_V6), BAN_WAIT, 0.5)
    check("a v6 source can be blocked in the kernel", took >= 0, f"after {took:.1f}s")
    ok, detail = tcp_probe(SERVER_V6, V6_PORT, timeout=3.0)
    check("the v6 DROP actually stops traffic", not ok, detail)
    check("operator unblock reports it undid something",
          inter.unblock_ip(CLIENT_V6) is True)
    check("the v6 DROP is gone", not drop_present("ip6tables", CLIENT_V6))
    ok, detail = tcp_probe(SERVER_V6, V6_PORT)
    check("and v6 flows again", ok, detail)


def phase_teardown(inter, chain: str, snap4: str, snap6: str) -> None:
    print("\n# stopping leaves the host exactly as it found it")
    stop(inter)
    st = inter.status()
    check("status reports not running after stop", st["running"] is False)
    check("nothing is left waiting for the kernel",
          st["pending_enforce"] == [] and st["pending_lift"] == [],
          f"enforce={st['pending_enforce']} lift={st['pending_lift']}")
    check("the v4 chain is gone", chain not in rules("iptables"), rules("iptables"))
    check("the INPUT jump is gone", f"-j {chain}" not in rules("iptables"))
    check("the v6 chain is gone", chain not in rules("ip6tables"))
    check("no NFQUEUE rule survives in either family",
          "NFQUEUE" not in rules("iptables") and "NFQUEUE" not in rules("ip6tables"))
    ok4, d4 = tcp_probe(SERVER_V4, TCP_PORT)
    ok6, d6 = tcp_probe(SERVER_V6, V6_PORT)
    check("v4 still connects with no consumer listening", ok4, d4)
    check("v6 still connects with no consumer listening", ok6, d6)
    check("host v4 ruleset matches the pre-run snapshot", rules("iptables") == snap4)
    check("host v6 ruleset matches the pre-run snapshot", rules("ip6tables") == snap6)


def nd_lladdr(peer: str, dev: str) -> str:
    """The client's cached link-layer address for *peer*, or "" if unresolved."""
    for line in in_client("ip", "-6", "neigh", "show", "dev", dev).stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == peer and "lladdr" in parts:
            return parts[parts.index("lladdr") + 1]
    return ""


def phase_icmp(inter, chain: str) -> None:
    print("\n# intercept_icmp: ICMP queued, and IPv6 has to survive it")
    before = inter.status()["nfqueue_packets"]
    rc = in_client("ping", "-c", "1", "-W", "2", SERVER_V4)
    last = (rc.stdout or "").strip().splitlines()
    check("echo-request passes while ICMP is queued", rc.returncode == 0,
          last[-1] if last else f"rc={rc.returncode}")
    queued = inter.status()["nfqueue_packets"] - before
    check("and it really was inspected", queued > 0, f"{queued} packets queued")

    # Neighbour discovery is IPv6's ARP, and it arrives as ICMPv6 (protocol 58)
    # from the peer: the solicitation that resolves *this* host's address is
    # exactly the packet the protocol gate used to block, which took the host
    # off the link the moment intercept_icmp was switched on.  Resolution is
    # read off the neighbour table rather than from ping6's exit status —
    # echo-request is a policy question this suite deliberately leaves blocked,
    # and a ping that fails for that reason would hide whether ND worked.
    sh("ip", "-6", "neigh", "flush", "dev", SERVER_DEV)
    in_client("ip", "-6", "neigh", "flush", "dev", CLIENT_DEV)
    check("the peer's neighbour cache starts empty",
          nd_lladdr(SERVER_V6, CLIENT_DEV) == "")
    ok, detail = tcp_probe(SERVER_V6, V6_PORT)
    check("neighbour discovery survives intercept_icmp",
          nd_lladdr(SERVER_V6, CLIENT_DEV) != "",
          f"lladdr={nd_lladdr(SERVER_V6, CLIENT_DEV) or 'unresolved'}")
    check("IPv6 service stays reachable under intercept_icmp", ok, detail)

    # The allowance covers link maintenance only.  Whether a stranger may ping
    # the host is still the operator's call: this engine allows ICMPv4 echo
    # (type 8) and nothing for ICMPv6, so a v6 echo has to keep failing — if it
    # stops, the gate is letting the whole of protocol 58 through.
    rc = in_client("ping", "-6", "-c", "1", "-W", "2", SERVER_V6, timeout=25)
    check("an ICMPv6 echo is still decided by policy", rc.returncode != 0,
          f"ping6 rc={rc.returncode}")
    check("neighbour traffic triggered no IPv6 ban",
          not drop_present("ip6tables", CLIENT_V6), rules("ip6tables"))


def preflight() -> str | None:
    """Return the reason to skip, or None when the suite can really run."""
    if sys.platform != "linux":
        return f"needs Linux (running on {sys.platform})"
    if not shutil.which("iptables"):
        return "iptables not installed"
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        return "needs root (run via sudo inside the server namespace)"
    try:
        import netfilterqueue  # noqa: F401
    except ImportError:
        return "the netfilterqueue binding is not installed"
    if CLIENT_NS not in sh("ip", "netns", "list"):
        return f"client namespace {CLIENT_NS} does not exist"
    # Binding is the check that cannot lie: it only succeeds when this
    # address really belongs to *this* namespace, which is the same thing the
    # suite is about to assume.  Parsing `ip addr` output would have depended
    # on the exact flags, and a wrong flag reads as an empty string — i.e. as
    # "wrong namespace" — and the suite would skip itself.
    probe = socket.socket(socket.AF_INET)
    try:
        probe.bind((SERVER_V4, 0))
    except OSError as exc:
        return f"{SERVER_V4} is not a local address here — run inside {SERVER_NS}: {exc}"
    finally:
        probe.close()
    return None


def main() -> int:
    skip = preflight()
    if skip:
        print(f"SKIP: {skip}")
        return 0

    from networksecurity.interception.iptables import IptablesManager

    chain = IptablesManager.CHAIN
    snap4, snap6 = rules("iptables"), rules("ip6tables")
    if re.search(rf"-N {chain}\b", snap4) or f"-A INPUT -j {chain}" in snap4:
        print(f"SKIP: chain {chain} already exists here, refusing to touch it")
        return 0

    listeners = Listeners()
    listeners.add(TCP_PORT)
    listeners.add(GUARDED_PORT)
    listeners.add(V6_PORT, socket.AF_INET6)

    tmpdir = Path("/tmp/nips-live")
    tmpdir.mkdir(exist_ok=True)

    try:
        phase_baseline(chain)

        verdicts: list[tuple[str, str, str]] = []
        rules_file = tmpdir / f"rules-{QUEUE_A}.json"
        rules_file.unlink(missing_ok=True)
        inter = build_interceptor(rules_file, QUEUE_A, False, verdicts)
        try:
            start(inter, QUEUE_A)
            phase_consumption(inter, chain, listeners, verdicts)
            phase_enforcement(inter, verdicts, rules_file)
            phase_ipv6(inter, chain, verdicts)
        finally:
            phase_teardown(inter, chain, snap4, snap6)

        # A second run with ICMP intercepted, on its own pipeline and queue
        # number: bans the first run escalated cannot colour what the ICMP
        # policy decides, and teardown is asserted twice.
        verdicts_b: list[tuple[str, str, str]] = []
        rules_file_b = tmpdir / f"rules-{QUEUE_B}.json"
        rules_file_b.unlink(missing_ok=True)
        inter_b = build_interceptor(rules_file_b, QUEUE_B, True, verdicts_b)
        try:
            start(inter_b, QUEUE_B)
            phase_icmp(inter_b, chain)
        finally:
            phase_teardown(inter_b, chain, snap4, snap6)
    finally:
        listeners.close()

    bad = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS, {len(bad)} FAIL")
    for name in bad:
        print(f"  failed: {name}")
    print(f"LIVE NFQUEUE E2E: {'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
