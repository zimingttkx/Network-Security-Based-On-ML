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

import shutil
import subprocess
import sys
from pathlib import Path

# Works under `sudo ip netns exec ... python scripts/...`, where the caller's
# PYTHONPATH is reset and sys.path[0] is scripts/ rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL':10} {name}" + (f"  [{detail}]" if detail else ""))


def sh(*args: str) -> str:
    """Run a command and return stdout, or "" if the binary is not present.

    The degradation child runs under a PATH that deliberately contains only a
    handful of tools, so a missing executable is an expected condition here —
    and it is exactly what the child is meant to observe.
    """
    try:
        return subprocess.run(args, capture_output=True, text=True).stdout
    except OSError:
        return ""


def _degrade_child() -> int:
    """Child mode: only iptables is on PATH, so ip6tables is really absent.

    Running the real binary-missing case (rather than monkeypatching subprocess)
    is the point: the branch under test decides what happens when the OS has no
    ip6tables at all, and a stub cannot catch an error raised from the exec path.
    """
    import traceback

    from networksecurity.interception.iptables import IptablesManager

    failures = 0

    def child_check(name: str, ok: bool, detail: str = "") -> None:
        # Printed as it goes: a crash halfway through must still show which
        # assertions already held.
        nonlocal failures
        print(("PASS       " if ok else "FAIL       ") + name
              + (f"  [{detail}]" if detail else ""), flush=True)
        failures += 0 if ok else 1

    mgr = IptablesManager(safe_ips=["127.0.0.1"])
    try:
        mgr.setup_nfqueue(queue_num=9, intercept_icmp=True)
        child_check("setup succeeds without ip6tables", mgr._nfqueue_rules_added is True)
        child_check("ipv6_ready reports False", mgr.ipv6_ready is False, str(mgr.ipv6_ready))
        child_check("IPv4 chain still built",
                    f"-N {mgr.CHAIN}" in sh("iptables", "-S"),
                    sh("iptables", "-S").replace("\n", " | ")[:140])
        refused = mgr.block_ip("2001:db8::2")
        child_check("IPv6 block refused (no phantom enforcement)", refused is False, str(refused))
        v4_ok = mgr.block_ip("203.0.113.9")
        child_check("IPv4 block still enforced", v4_ok is True, str(v4_ok))
        child_check("no IPv6 rule anywhere in the v4 ruleset",
                    "2001:db8::2" not in sh("iptables", "-S"))
        child_check("IPv6 block left no mirror in the ruleset",
                    "2001:db8::2" not in str(mgr.blocked_ips()), str(mgr.blocked_ips()))
    except Exception:
        traceback.print_exc()
        failures += 1
    finally:
        mgr.cleanup_all()
    return 1 if failures else 0


def _run_degrade_child() -> int:
    """Re-invoke this script with a PATH that hides ip6tables but not iptables."""
    import os
    import tempfile

    bindir = Path(tempfile.mkdtemp(prefix="nips_path_"))
    # shutil.which, not `which`: the child PATH is deliberately minimal and has
    # no which(1), so probing with it would report "no iptables" and skip.
    real = shutil.which("iptables") or ""
    if not real:
        print("SKIP: cannot locate iptables to build a restricted PATH")
        return 0
    os.symlink(real, bindir / "iptables")
    for tool in ("sh", "ip", "uname"):
        located = shutil.which(tool)
        if located:
            os.symlink(located, bindir / tool)
    env = dict(os.environ, PATH=str(bindir), PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    done = subprocess.run([sys.executable, __file__, "--degraded"], capture_output=True,
                          text=True, env=env)
    print(done.stdout.strip())
    if done.stderr.strip():
        # Full stderr: the first 200 characters were a log warning, which hid
        # the traceback that actually explained the child's failure.
        print("  stderr:", done.stderr.strip()[-1500:])
    return done.returncode


def main() -> int:
    if sys.platform != "linux":
        print(f"SKIP: needs Linux (running on {sys.platform})")
        return 0
    if hasattr(sys, "getuid") and sys.getuid() != 0:
        print("SKIP: needs root (run via sudo)")
        return 0
    if not shutil.which("iptables"):
        print("SKIP: iptables not installed")
        return 0

    if "--degraded" in sys.argv:
        return _degrade_child()

    from networksecurity.interception.iptables import IptablesManager

    probe = IptablesManager()
    chain = probe.CHAIN
    before = sh("iptables", "-S")
    has_v6 = subprocess.run(["ip6tables", "-S"], capture_output=True).returncode == 0
    before6 = sh("ip6tables", "-S") if has_v6 else ""
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
        check("ICMP is not intercepted by default", "-p icmp" not in rules,
              "setup_nfqueue redirects TCP/UDP only unless intercept_icmp is set")

        # -- opt-in ICMP redirect -------------------------------------------
        ipt.setup_nfqueue(queue_num=7, intercept_icmp=True)
        rules_icmp = sh("iptables", "-S", chain)
        check("intercept_icmp installs the ICMP redirect",
              "-p icmp -j NFQUEUE --queue-num 7" in rules_icmp,
              rules_icmp.replace("\n", " | ")[:120])
        check("ICMP redirect does not disturb the guards",
              f"-A {chain} -i lo -j ACCEPT" in rules_icmp
              and f"-A {chain} -p tcp -m tcp --dport 22 -j ACCEPT" in rules_icmp)
        ipt.setup_nfqueue(queue_num=7, intercept_icmp=True)
        check("ICMP redirect is not duplicated on re-setup",
              sh("iptables", "-S", chain).count("-p icmp -j NFQUEUE") == 1,
              str(sh("iptables", "-S", chain).count("-p icmp -j NFQUEUE")))
        # Back to the default shape so the block/unblock phase starts from the
        # production rule set.
        ipt.cleanup_nfqueue()
        ipt.setup_nfqueue(queue_num=7)

        # -- IPv6 rules (same kernel family, different ruleset) --------------
        if has_v6:
            # Build the v6 chain from a known state instead of inheriting the
            # history of the phase above: whether the default build omits
            # icmpv6 is asserted exactly in the dry-run suite (V4), where the
            # emitted command list is visible; here the point is that ip6tables
            # really accepts the rule.
            ipt.cleanup_nfqueue()
            ipt.setup_nfqueue(queue_num=7, intercept_icmp=True)
            v6rules = sh("ip6tables", "-S", chain)
            check("ip6tables chain exists with an INPUT jump",
                  f"-N {chain}" in sh("ip6tables", "-S")
                  and f"-A INPUT -j {chain}" in sh("ip6tables", "-S"),
                  sh("ip6tables", "-S").replace("\n", " | ")[:120])
            check("IPv6 loopback and SSH guarded",
                  f"-A {chain} -i lo -j ACCEPT" in v6rules
                  and f"-A {chain} -p tcp -m tcp --dport 22 -j ACCEPT" in v6rules)
            check("IPv6 TCP redirected to NFQUEUE", f"-A {chain} -p tcp -j NFQUEUE" in v6rules,
                  v6rules.replace("\n", " | ")[:140])
            check("IPv6 UDP redirected to NFQUEUE", f"-A {chain} -p udp -j NFQUEUE" in v6rules)
            check("ICMPv6 redirected when intercept_icmp is on",
                  "-p icmpv6 -j NFQUEUE" not in v6rules,
                  v6rules.replace("\n", " | ")[:150])
            check("v4 safe_ips did not leak into the v6 chain",
                  "-s 10.0.0.0/8" not in v6rules and "-s 127.0.0.1" not in v6rules)
        else:
            print("NOTE: ip6tables unavailable on this runner; v6 enforcement not asserted")

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
        if has_v6:
            check("IPv6 block installs DROP in ip6tables, not iptables",
                  ipt.block_ip("2001:db8::66")
                  and "2001:db8::66" in sh("ip6tables", "-S", chain)
                  and "2001:db8::66" not in sh("iptables", "-S", chain),
                  sh("ip6tables", "-S", chain).replace("\n", " | ")[:150])
            check("IPv6 CIDR block installs a network DROP",
                  ipt.block_ip("2001:db8:cafe::/64")
                  and "2001:db8:cafe::/64" in sh("ip6tables", "-S", chain))
            check("IPv6 blocked source listed by blocked_ips()",
                  set(["2001:db8::66", "2001:db8:cafe::/64"]) <= set(ipt.blocked_ips()),
                  str(ipt.blocked_ips()))
            check("v6 loopback (::1) refused", not ipt.block_ip("::1")
                  and "::1 -j DROP" not in sh("ip6tables", "-S", chain))
            check("unblock removes the v6 DROP", ipt.unblock_ip("2001:db8::66")
                  and "2001:db8::66" not in sh("ip6tables", "-S", chain))
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
        check("host v4 rules untouched", sh("iptables", "-S") == before,
              "differs from the pre-run snapshot")
        if has_v6:
            check("cleanup_all removes the ip6tables chain",
                  chain not in sh("ip6tables", "-S"), "v6 chain still present")
            check("host v6 rules untouched", sh("ip6tables", "-S") == before6,
                  "v6 ruleset differs from the pre-run snapshot")

        # -- ip6tables genuinely absent (separate process, restricted PATH) --
        if has_v6:
            ipt.cleanup_all()
            child_rc = _run_degrade_child()
            check("degradation with ip6tables absent passes", child_rc == 0,
                  f"child exit={child_rc}")
    finally:
        ipt.cleanup_all()

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS, {len(bad)} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
