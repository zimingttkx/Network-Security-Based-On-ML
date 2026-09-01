#!/usr/bin/env python3
"""Cross-validation for interception/ module (offline-safe parts).

Live nfqueue capture cannot run without root; these checks cover:
- PacketParser.from_raw byte-level correctness against hand-built packets
- PacketParser.from_dict defaults
- Interceptor state machine without root (setup() must raise, not half-start)
- fail-closed callback contract of NFQueueHandler (callback-less => accept? no:
  callback None means should_drop False -> ACCEPT; verify documented behavior)
- IptablesManager rule-construction dry-run via command recording
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
                     flags=0x02, payload=b"hello", ttl=64, ip_total=None):
    """Hand-build a minimal IPv4+TCP packet."""
    tcp_hdr = struct.pack("!HHIIBBHHH", sport, dport, 1, 1, (5 << 4), flags, 8192, 0, 0)
    tcp = tcp_hdr + payload
    total = 20 + len(tcp) if ip_total is None else ip_total
    ip_hdr = struct.pack("!BBHHHBBH4s4s",
                         0x45, 0, total, 1, 0, ttl, 6, 0,
                         bytes(map(int, src.split("."))),
                         bytes(map(int, dst.split("."))))
    return ip_hdr + tcp


def build_udp_packet(src="10.0.0.1", dst="10.0.0.2", sport=5000, dport=53, payload=b"dn"):
    udp = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    ip_hdr = struct.pack("!BBHHHBBH4s4s",
                         0x45, 0, 20 + len(udp), 1, 0, 64, 17, 0,
                         bytes(map(int, src.split("."))),
                         bytes(map(int, dst.split("."))))
    return ip_hdr + udp


# --- I1: TCP parse -----------------------------------------------------------
raw = build_tcp_packet()
p = PacketParser.from_raw(raw, timestamp=123.0)
report("I1 TCP parse fields",
       not (p and p.src_ip == "10.0.0.1" and p.dst_port == 80 and p.protocol == 6
            and p.tcp_flags == 0x02 and p.payload_size == 5 and p.packet_size == 45
            and p.ttl == 64),
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
p = PacketParser.from_raw(raw, timestamp=1.0)
raw_trunc_tcp = raw[:20 + 10]  # IP header + 10 bytes of TCP header (no full TCP hdr)
p = PacketParser.from_raw(raw_trunc_tcp, timestamp=1.0)
report("I4b truncated TCP: ports default 0, still parsed",
       not (p is not None and p.src_port == 0 and p.dst_port == 0 and p.protocol == 6),
       f"{p}")

# --- I5: payload_size clamp (total_len lie) -----------------------------------
raw = build_tcp_packet(ip_total=10)  # IP total length smaller than header
p = PacketParser.from_raw(raw, timestamp=1.0)
report("I5 lying total_len clamps payload>=0",
       not (p is not None and p.payload_size == 0), f"{p}")

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
    mgr.block_ip("6.6.6.6")
    mgr.unblock_ip("6.6.6.6")
    mgr.cleanup_all()
    joined = [" ".join(c) for c in commands]
    has_chain = any("-N NIPS" in c for c in joined)
    has_jump = any("-I INPUT -j NIPS" in c for c in joined)
    has_tcp_nfq = any("NFQUEUE --queue-num 5" in c and "tcp" in c for c in joined)
    has_udp_nfq = any("NFQUEUE --queue-num 5" in c and "udp" in c for c in joined)
    has_block = any("-I NIPS 1 -s 6.6.6.6 -j DROP" in c for c in joined)
    has_unblock = any("-D NIPS -s 6.6.6.6 -j DROP" in c for c in joined)
    has_flush = any("-F NIPS" in c and "-X" in " ".join(joined) for c in joined)
    ok = all([has_chain, has_jump, has_tcp_nfq, has_udp_nfq, has_block, has_unblock])
    report("I9 iptables rule construction", not ok,
           f"chain={has_chain} jump={has_jump} tcp={has_tcp_nfq} udp={has_udp_nfq} "
           f"block={has_block} unblock={has_unblock}")
    ipv6_safe = any("ip6tables -I NIPS -s ::1 -j ACCEPT" in c for c in joined)
    report("I10 IPv6 safe_ip routed to ip6tables", not ipv6_safe, f"ip6tables rule present={ipv6_safe}")
finally:
    ipt_mod.subprocess.run = real_run

print("\n==== SUMMARY ====")
for name, status in results:
    print(f"  {status:14s} {name}")
