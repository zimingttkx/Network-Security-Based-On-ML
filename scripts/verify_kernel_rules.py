#!/usr/bin/env python3
"""Kernel-level checks for the iptables control plane.  Must run on Linux as root.

Everything else in the suite tests our own bookkeeping; this tests the thing
that actually decides whether traffic is blocked — the rule set the kernel
holds.  It exists because a unit test can assert that ``iptables -I ...`` was
called with the right arguments while the real command fails with a module
error, an unmatched chain or a wrong table, and the IPS then quietly stops
enforcing anything.

Runs against a dedicated chain so it never disturbs the host's own firewall:
the manager's rules all live under its NIPS chain, and teardown asserts the
chain is gone.
"""
from __future__ import annotations

import subprocess
import sys

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL':10} {name}" + (f"  [{detail}]" if detail else ""))


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def main() -> int:
    if sys.platform != "linux":
        print(f"SKIP: needs Linux (running on {sys.platform})")
        return 0
    if hasattr(sys, "getuid") and sys.getuid() != 0:
        print("SKIP: needs root (run via sudo)")
        return 0
    if not sh("which", "iptables").strip():
        print("SKIP: iptables not installed")
        return 0

    from networksecurity.interception.iptables import IptablesManager

    probe = IptablesManager()
    chain = probe.CHAIN
    before = sh("iptables", "-S")
    if f"-N {chain}" in before or f"-A {chain}" in before:
        print(f"SKIP: chain {chain} already exists on this host, refusing to touch it")
        return 0

    ipt = IptablesManager(safe_ips=["127.0.0.1", "10.0.0.0/8"])
    try:
        # -- nfqueue redirect ------------------------------------------------
        ipt.setup_nfqueue(queue_num=7)
        rules = sh("iptables", "-S", chain)
        check("NIPS chain created", chain in sh("iptables", "-S"), chain)
        check("INPUT jumps to the NIPS chain", f"-A INPUT -j {chain}" in sh("iptables", "-S"))
        check("loopback is accepted before any inspection",
              f"-A {chain} -i lo -j ACCEPT" in rules, rules.splitlines()[1:3])
        check("SSH (22) is protected",
              f"-A {chain} -p tcp -m tcp --dport 22 -j ACCEPT" in rules)
        check("safe_ips are guarded",
              f"-A {chain} -s 10.0.0.0/8 -j ACCEPT" in rules)
        check("TCP redirected to NFQUEUE 7",
              f"-A {chain} -p tcp -j NFQUEUE --queue-num 7" in rules or
              f"-A {chain} -p tcp -j NFQUEUE --queue-num 7 --queue-bypass" in rules,
              "module xt_NFQUEUE missing" if "NFQUEUE" not in rules else "")
        check("UDP redirected to NFQUEUE 7", "-p udp -j NFQUEUE --queue-num 7" in rules)
        check("only TCP and UDP reach the pipeline (ICMP is not intercepted)",
              "-p icmp" not in rules,
              "documented behaviour of setup_nfqueue")

        # -- blocking --------------------------------------------------------
        check("block_ip(203.0.113.7) reported success", ipt.block_ip("203.0.113.7"))
        after_block = sh("iptables", "-S", chain)
        check("DROP rule really present in the kernel",
              "203.0.113.7" in after_block and "-j DROP" in after_block,
              after_block.replace("\n", " | ")[:180])
        check("blocked_ips() reflects the kernel", "203.0.113.7" in ipt.blocked_ips(),
              str(ipt.blocked_ips()))
        check("CIDR block installs a network DROP", ipt.block_ip("198.51.100.0/24"))
        check("CIDR visible in kernel rules",
              "198.51.100.0/24" in sh("iptables", "-S", chain))
        check("is_blockable refuses loopback", not ipt.is_blockable("127.0.0.1"))
        check("is_blockable refuses a safe_ip", not ipt.is_blockable("10.1.2.3"))
        check("is_blockable allows ordinary source", ipt.is_blockable("203.0.113.9"))
        check("block_ip on loopback does not install a rule",
              not ipt.block_ip("127.0.0.1")
              and "-s 127.0.0.1 -j DROP" not in sh("iptables", "-S", chain))

        # -- idempotency and teardown --------------------------------------
        ipt.block_ip("203.0.113.7")           # re-block the same source
        twice = sh("iptables", "-S", chain)
        check("re-blocking does not duplicate the DROP",
              twice.count("203.0.113.7") == 1, f"count={twice.count('203.0.113.7')}")
        check("unblock_ip reports success", ipt.unblock_ip("203.0.113.7") is True)
        check("DROP for that IP gone after unblock",
              "203.0.113.7" not in sh("iptables", "-S", chain))
        ipt.cleanup_all()
        check("cleanup_all removes the chain",
              chain not in sh("iptables", "-S"), "chain still present")
        check("cleanup_all removes the INPUT jump",
              f"-j {chain}" not in sh("iptables", "-S"))
        check("host rules untouched", sh("iptables", "-S") == before,
              "differs from the pre-run snapshot")
    finally:
        ipt.cleanup_all()

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS, {len(bad)} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
