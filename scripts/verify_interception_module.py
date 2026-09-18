#!/usr/bin/env python3
"""Cross-validation for interception/ module (offline-safe parts).

Live nfqueue capture cannot run without root; these checks cover:
- PacketParser.from_raw byte-level correctness against hand-built packets
- PacketParser.from_raw fail-closed rejection of malformed/inconsistent headers
- PacketParser.from_dict defaults
- Interceptor state machine without root (setup() must raise, not half-start)
- NFQueueHandler fail-closed callback contract, driven offline via
  _handle_packet (netfilterqueue is imported lazily inside start()): no
  callback, unparseable packet and cooperative stop must all DROP, never
  accept
- IptablesManager rule-construction dry-run via command recording

Exit status is 1 if any check reports CONFIRMED-BUG, 0 otherwise.
"""
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networksecurity.interception.packet_parser import PacketParser

results = []


def report(name: str, confirmed: bool, evidence: str):
    status = "CONFIRMED-BUG" if confirmed else "PASS"
    results.append((name, status))
    print(f"[{status}] {name}\n        {evidence}\n", flush=True)


def build_tcp_packet(src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=80,
                     flags=0x02, payload=b"hello", ttl=64, ip_total=None,
                     version_ihl=0x45, frag=0, tcp_dataofs=5, window=8192):
    """Hand-build a minimal IPv4+TCP packet.

    ``version_ihl``/``frag``/``tcp_dataofs`` exist so the malformed-header
    cases (I11-I14) can inject a specific lie into an otherwise valid packet.
    """
    tcp_hdr = struct.pack("!HHIIBBHHH", sport, dport, 1, 1,
                          (tcp_dataofs << 4), flags, window, 0, 0)
    tcp = tcp_hdr + payload
    total = 20 + len(tcp) if ip_total is None else ip_total
    ip_hdr = struct.pack("!BBHHHBBH4s4s",
                         version_ihl, 0, total, 1, frag, ttl, 6, 0,
                         bytes(map(int, src.split("."))),
                         bytes(map(int, dst.split("."))))
    return ip_hdr + tcp


def build_udp_packet(src="10.0.0.1", dst="10.0.0.2", sport=5000, dport=53,
                     payload=b"dn", ip_total=None):
    udp = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    total = 20 + len(udp) if ip_total is None else ip_total
    ip_hdr = struct.pack("!BBHHHBBH4s4s",
                         0x45, 0, total, 1, 0, 64, 17, 0,
                         bytes(map(int, src.split("."))),
                         bytes(map(int, dst.split("."))))
    return ip_hdr + udp


def with_proto(raw, proto):
    """Rewrite the IP protocol byte of an already-built packet.

    The IP-header guards (IHL bounds, total_len < ihl) are otherwise masked by
    the TCP/UDP-specific length checks: a malformed TCP packet is refused by
    those even with the IP-level guard deleted.  Carrying the same malformed
    header under a protocol the parser does not dissect (ICMP) isolates them.
    """
    return raw[:9] + bytes([proto]) + raw[10:]


# --- I1: TCP parse -----------------------------------------------------------
raw = build_tcp_packet()
p = PacketParser.from_raw(raw, timestamp=123.0)
report("I1 TCP parse fields",
       not (p and p.src_ip == "10.0.0.1" and p.dst_port == 80 and p.protocol == 6
            and p.tcp_flags == 0x02 and p.payload_size == 5 and p.packet_size == 45
            and p.ttl == 64 and p.window_size == 8192 and p.direction == 1),
       f"{p}")

# --- I2: UDP parse ------------------------------------------------------------
raw = build_udp_packet()
p = PacketParser.from_raw(raw, timestamp=124.0)
report("I2 UDP parse fields",
       not (p and p.protocol == 17 and p.src_port == 5000 and p.dst_port == 53
            and p.payload_size == 2),
       f"{p}")

# --- I3: non-IPv4 rejected ----------------------------------------------------
p = PacketParser.from_raw(b"\x60" + b"\x00" * 40, timestamp=1.0)  # IPv6 version nibble
report("I3 IPv6 rejected (None)", p is not None, f"returns {p}")

# --- I4: truncated packets ----------------------------------------------------
raw = build_tcp_packet()
p = PacketParser.from_raw(raw[:10], timestamp=1.0)
report("I4a truncated (<20B) rejected", p is not None, f"returns {p}")
raw_trunc_tcp = raw[:20 + 10]  # IP header + 10 bytes of TCP header (no full TCP hdr)
p = PacketParser.from_raw(raw_trunc_tcp, timestamp=1.0)
report("I4b truncated TCP header rejected (None, not ports=0)",
       p is not None,
       f"returns {p} — an incomplete TCP header used to yield a PacketInfo with "
       f"src_port/dst_port defaulted to 0, which the whole detection chain then "
       f"treated as a real port")

# --- I5: payload_size clamp (total_len lie) -----------------------------------
raw = build_tcp_packet(ip_total=10)  # IP total length smaller than header
p = PacketParser.from_raw(raw, timestamp=1.0)
report("I5 lying total_len rejected (None)", p is not None,
       f"returns {p} — total_len=10 < ihl=20 used to be clamped into a "
       f"payload_size=0 PacketInfo instead of being refused")
# Same lie on a protocol the parser does not dissect: the TCP/UDP length checks
# would refuse I5 on their own, so this is what pins the IP-level guard.
p = PacketParser.from_raw(with_proto(build_tcp_packet(ip_total=10), 1), timestamp=1.0)
report("I5b ICMP total_len=10 < ihl rejected", p is not None,
       f"returns {p} — without the total_len<ihl guard a protocol-1 packet "
       f"parses with payload_size clamped to 0")

# --- I6: from_dict defaults ----------------------------------------------------
p = PacketParser.from_dict({})
report("I6 from_dict defaults",
       not (p.src_ip == "0.0.0.0" and p.protocol == 6 and p.ttl == 64), f"{p}")

# --- I7: timestamp handling (0 vs None) ----------------------------------------
raw = build_tcp_packet()
p = PacketParser.from_raw(raw, timestamp=0.0)
report("I7 zero timestamp preserved", p.timestamp != 0.0, f"timestamp={p.timestamp}")

# --- I8: Interceptor rejects non-root ------------------------------------------
try:
    import os
    if os.geteuid() == 0:
        print("[SKIP] I8 running as root — non-root path untestable here")
    else:
        from networksecurity.interception import Interceptor
        from networksecurity.engine import DetectionPipeline
        it = Interceptor(DetectionPipeline())
        try:
            it.setup()
            report("I8 setup() as non-root raises", False, "setup() did not raise")
        except RuntimeError as e:
            report("I8 setup() as non-root raises", "root" not in str(e).lower(),
                   f"RuntimeError: {e}")
except Exception as e:  # noqa: BLE001
    report("I8 interceptor import", True, f"{type(e).__name__}: {e}")

# --- I11: IHL outside [20,60] or beyond the captured bytes ---------------------
# Carried under ICMP: with protocol 6/17 the L4 length checks would refuse the
# same packet on their own and the IHL guard would go untested.
p = PacketParser.from_raw(with_proto(build_tcp_packet(version_ihl=0x42), 1),
                          timestamp=1.0)
report("I11a ihl=8 (< 20) rejected", p is not None,
       f"returns {p} — ihl=8 puts every L4 offset inside the IP header, and "
       f"src/dst would be read from the wrong bytes")
# ihl=60 but only 45 bytes captured: the header claims options that are not there.
p = PacketParser.from_raw(with_proto(build_tcp_packet(version_ihl=0x4F), 1),
                          timestamp=1.0)
report("I11b ihl=60 > len(data) rejected", p is not None,
       f"returns {p} — ihl=60 with 45 captured bytes used to parse as a "
       f"protocol-1 packet with payload_size clamped to 0")
p = PacketParser.from_raw(build_tcp_packet(version_ihl=0x42), timestamp=1.0)
report("I11c ihl=8 on TCP rejected", p is not None, f"returns {p}")
# Deliberate contract: a *well-formed* IPv4 packet whose protocol we do not
# dissect is still returned with ports at 0, so the rule engine's
# allowed_protocols filter blocks it — an auditable outcome rather than a
# silent parse failure that would be indistinguishable from a malformed frame.
p = PacketParser.from_raw(with_proto(build_tcp_packet(), 1), timestamp=1.0)
report("I11d well-formed ICMP still parsed (ports 0)",
       not (p is not None and p.protocol == 1 and p.src_port == 0
            and p.dst_port == 0 and p.payload_size == 25),
       f"{p}")

# --- I12: fragments -------------------------------------------------------------
# frag=1 -> non-first fragment: the bytes at `ihl` are a continuation of an
# earlier fragment's payload, not an L4 header.
p = PacketParser.from_raw(build_tcp_packet(frag=1), timestamp=1.0)
report("I12a non-first fragment rejected", p is not None, f"returns {p}")
# frag=0x2000 -> MF set, offset 0: this IS the first fragment and does carry an
# L4 header, so it must still parse.
p = PacketParser.from_raw(build_tcp_packet(frag=0x2000), timestamp=1.0)
report("I12b first fragment (MF only) still parsed",
       not (p is not None and p.src_port == 1234 and p.dst_port == 80), f"{p}")

# --- I13: TCP data offset outside [20,60] ----------------------------------------
p = PacketParser.from_raw(build_tcp_packet(tcp_dataofs=0), timestamp=1.0)
report("I13a data_offset=0 rejected", p is not None, f"returns {p}")
p = PacketParser.from_raw(build_tcp_packet(tcp_dataofs=1), timestamp=1.0)
report("I13b data_offset=4 rejected", p is not None, f"returns {p}")
# dataofs=15 claims a 60-byte TCP header; only 25 bytes are present.
p = PacketParser.from_raw(build_tcp_packet(tcp_dataofs=15), timestamp=1.0)
report("I13c data_offset=60 with 25 bytes captured rejected", p is not None,
       f"returns {p}")

# --- I14: total_len inconsistent with the headers --------------------------------
p = PacketParser.from_raw(build_tcp_packet(ip_total=30), timestamp=1.0)
report("I14a TCP total_len=30 < ihl+20 rejected", p is not None, f"returns {p}")
p = PacketParser.from_raw(build_udp_packet(ip_total=24), timestamp=1.0)
report("I14b UDP total_len=24 < ihl+8 rejected", p is not None, f"returns {p}")

# --- I15: TCP options honoured, advertised window parsed --------------------------
raw = build_tcp_packet(payload=b"hello", tcp_dataofs=6, window=65535, ip_total=49)
raw = raw[:40] + b"\x01\x01\x08\x0a" + raw[40:]  # splice 4 option bytes in
p = PacketParser.from_raw(raw, timestamp=1.0)
report("I15 window_size parsed, payload_size uses data_offset not 20",
       not (p is not None and p.window_size == 65535 and p.payload_size == 5
            and p.packet_size == 49),
       f"{p}")

# --- I16-I18: NFQueueHandler fail-closed callback contract -----------------------
# netfilterqueue is imported lazily inside start(), so the handler can be
# exercised offline by driving _handle_packet with a stand-in nf_packet.
from networksecurity.interception.nfqueue_handler import NFQueueHandler


class FakeNfPacket:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.verdict = None

    def get_payload(self):
        return self._payload

    def drop(self):
        self.verdict = "drop"

    def accept(self):
        self.verdict = "accept"


class FakeQueue:
    def __init__(self):
        self.unbound = 0

    def unbind(self):
        self.unbound += 1


good_tcp = build_tcp_packet()

h = NFQueueHandler(queue_num=7)
pkt = FakeNfPacket(good_tcp)
h._handle_packet(pkt)
report("I16 no callback set -> drop (not accept)",
       not (pkt.verdict == "drop" and h.dropped_count == 1
            and h.parse_failed_count == 0),
       f"verdict={pkt.verdict} status={h.status()}")

h = NFQueueHandler(queue_num=7)
h.set_callback(lambda info: False)
pkt = FakeNfPacket(b"\x60" + b"\x00" * 40)  # IPv6 -> unparseable
h._handle_packet(pkt)
report("I17 unparseable packet -> drop + parse_failed_count",
       not (pkt.verdict == "drop" and h.parse_failed_count == 1),
       f"verdict={pkt.verdict} status={h.status()}")

h = NFQueueHandler(queue_num=7)
h.set_callback(lambda info: False)
fake_queue = FakeQueue()
h._queue = fake_queue
h._stop_requested = True
pkt = FakeNfPacket(good_tcp)
h._handle_packet(pkt)
report("I18 cooperative stop -> unbind on loop thread + drop",
       not (pkt.verdict == "drop" and fake_queue.unbound == 1 and h._queue is None),
       f"verdict={pkt.verdict} unbind_calls={fake_queue.unbound} "
       f"queue_cleared={h._queue is None}")

# --- I19: live timestamps are monotonic, not wall-clock ------------------------
import time as _time

seen = []
h = NFQueueHandler(queue_num=7)
h.set_callback(seen.append)
before = _time.monotonic()
h._handle_packet(FakeNfPacket(good_tcp))
after = _time.monotonic()
ts = seen[0].timestamp if seen else None
report("I19 capture timestamp from time.monotonic()",
       not (ts is not None and before - 1 <= ts <= after + 1),
       f"timestamp={ts} monotonic window=[{before}, {after}] — a wall-clock source "
       f"lets an NTP step reset rate-limit windows and corrupt AfterImage decay")

# --- I20: a failed bind must not leave the handler "started" -------------------
import networksecurity.interception.nfqueue_handler as nfq_mod


class FailingQueue:
    def bind(self, queue_num, cb):
        raise OSError("netlink bind failed")


class FakeNfqModule:
    NetfilterQueue = FailingQueue


real_get_nfqueue = nfq_mod._get_nfqueue
nfq_mod._get_nfqueue = lambda: FakeNfqModule
try:
    h = NFQueueHandler(queue_num=7)
    raised = False
    try:
        h.start()
    except OSError:
        raised = True
    report("I20 failed bind leaves running=False",
           not (raised and h.status()["running"] is False),
           f"raised={raised} status={h.status()} — setting _running before bind "
           f"made every later start() return immediately while the kernel "
           f"delivered nothing")
finally:
    nfq_mod._get_nfqueue = real_get_nfqueue

# --- I24: setup() rolls back when a late step fails ------------------------------
import os
import shutil

from networksecurity.engine import DetectionPipeline
from networksecurity.interception import Interceptor


class ExplodingIptables:
    """setup_nfqueue fails after the event loop thread is already running."""

    def __init__(self):
        self.cleaned = False

    def setup_nfqueue(self, queue_num: int = 0) -> None:
        raise RuntimeError("iptables rejected the redirect")

    def cleanup_all(self) -> None:
        self.cleaned = True


it24 = Interceptor(DetectionPipeline())
stub24 = ExplodingIptables()
it24._iptables = stub24
_real_geteuid, _real_which = os.geteuid, shutil.which
os.geteuid = lambda: 0
shutil.which = lambda name: "/usr/sbin/" + name
try:
    raised24 = False
    try:
        it24.setup()
    except RuntimeError:
        raised24 = True
finally:
    os.geteuid, shutil.which = _real_geteuid, _real_which
report("I24 setup() failure rolls the half-initialised interceptor back",
       not (raised24 and it24._loop is None and it24._loop_thread is None
            and it24.running is False and stub24.cleaned),
       f"raised={raised24} loop={it24._loop} thread={it24._loop_thread} "
       f"running={it24.running} cleanup_all={stub24.cleaned} — without the "
       f"rollback a live kernel redirect can be left with no listener "
       f"draining the queue, stalling every matched packet")

# --- I9: iptables manager command dry-run --------------------------------------
# Monkeypatch subprocess.run to record commands instead of executing.
import networksecurity.interception.iptables as ipt_mod


class FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


commands = []


def fake_run(args, **kwargs):
    commands.append(args)
    return FakeCompleted()


real_run = ipt_mod.subprocess.run
ipt_mod.subprocess.run = fake_run
try:
    # Fresh instance simulating empty firewall (rule_exists -> False)
    ipt_mod.subprocess.run = fake_run  # -C probes also recorded & return rc0…
    # Simulate -C probes failing (rule absent): return nonzero for -C
    def fake_run2(args, **kwargs):
        commands.append(args)
        fc = FakeCompleted()
        if args[1] == "-C":
            fc.returncode = 1
        return fc
    ipt_mod.subprocess.run = fake_run2

    mgr = ipt_mod.IptablesManager(safe_ips=["127.0.0.1", "::1"])
    mgr.setup_nfqueue(queue_num=5)
    guard_count = mgr._guard_rule_count
    blocked_ok = mgr.block_ip("6.6.6.6")
    unblocked_ok = mgr.unblock_ip("6.6.6.6")
    mgr.cleanup_all()
    joined = [" ".join(c) for c in commands]
    has_chain = any("-N NIPS" in c for c in joined)
    has_jump = any("-I INPUT -j NIPS" in c for c in joined)
    has_tcp_nfq = any("NFQUEUE --queue-num 5" in c and "tcp" in c for c in joined)
    has_udp_nfq = any("NFQUEUE --queue-num 5" in c and "udp" in c for c in joined)
    # The DROP must land *below* the ACCEPT guards.  safe_ips=["127.0.0.1","::1"]
    # yields 3 IPv4 guards (127.0.0.1, -i lo, dport 22); "::1" goes to ip6tables
    # and is not part of the IPv4 chain.  Position 1 — the pre-fix behaviour —
    # put every ban over the rules whose whole purpose is to keep the box
    # reachable, self-DoSing SSH and loopback the moment anything tripped a
    # threshold.
    drop_pos = guard_count + 1
    has_block = any(
        f"-I NIPS {drop_pos} -s 6.6.6.6 -j DROP" in c for c in joined)
    has_unblock = any("-D NIPS -s 6.6.6.6 -j DROP" in c for c in joined)
    ok = all([has_chain, has_jump, has_tcp_nfq, has_udp_nfq,
              has_block, has_unblock, blocked_ok, unblocked_ok])
    report("I9 iptables rule construction", not ok,
           f"chain={has_chain} jump={has_jump} tcp={has_tcp_nfq} udp={has_udp_nfq} "
           f"block@{drop_pos}={has_block} unblock={has_unblock} "
           f"block_ret={blocked_ok} unblock_ret={unblocked_ok}")
    ipv6_safe = any("ip6tables -I NIPS -s ::1 -j ACCEPT" in c for c in joined)
    report("I10 IPv6 safe_ip routed to ip6tables", not ipv6_safe, f"ip6tables rule present={ipv6_safe}")

    # Guards are all inserted before the NFQUEUE redirects are appended, so a
    # DROP at guard_count+1 also sits above the queue rules — enforced traffic
    # never reaches userspace detection.
    report("I23 guard count matches the 3 IPv4 ACCEPT guards",
           guard_count != 3,
           f"guard_count={guard_count} — a miscount shifts every later DROP "
           f"over a guard or under the NFQUEUE redirects")

    commands.clear()
    mgr2 = ipt_mod.IptablesManager(safe_ips=["127.0.0.1", "10.9.8.0/24"])
    mgr2.setup_nfqueue(queue_num=5)
    refused_loop = mgr2.block_ip("127.0.0.53")
    refused_safe = mgr2.block_ip("127.0.0.1")
    refused_cidr = mgr2.block_ip("10.9.8.7")
    accepted = mgr2.block_ip("203.0.113.66")
    idempotent = mgr2.block_ip("203.0.113.66")
    report("I21 block_ip returns False for refused sources, True for real ones",
           refused_loop or refused_safe or refused_cidr or not accepted or not idempotent,
           f"loopback={refused_loop} safe_ip={refused_safe} safe_cidr={refused_cidr} "
           f"remote={accepted} repeat={idempotent} — a True from a refused block "
           f"is what let phantom blacklist entries into rules.json")

    never_blocked = mgr2.unblock_ip("198.51.100.1")
    real_unblock = mgr2.unblock_ip("203.0.113.66")
    second_unblock = mgr2.unblock_ip("203.0.113.66")
    report("I22 unblock_ip reports whether the kernel rule really went away",
           never_blocked or not real_unblock or second_unblock,
           f"never_blocked={never_blocked} after_block={real_unblock} "
           f"repeat={second_unblock} — a caller that trusts a False positive "
           f"drops its mirror while the DROP is still installed")
finally:
    ipt_mod.subprocess.run = real_run

print("\n==== SUMMARY ====")
for name, status in results:
    print(f"  {status:14s} {name}")

failures = [name for name, status in results if status != "PASS"]
print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
sys.exit(1 if failures else 0)
