"""iptables / nftables rule management (Linux only)."""

from __future__ import annotations

import logging
import subprocess
import threading
from ipaddress import ip_network

logger = logging.getLogger(__name__)


def blockable(ip: str, safe_ips=()) -> bool:
    """True when a kernel DROP may be installed for this source.

    Loopback sources are definitionally local traffic — this host, its DNS
    stub resolver (127.0.0.53), or an internal service.  Blocking any of
    them is a self-DoS (observed live: the systemd-resolved stub got a
    kernel DROP, silently killing host DNS), and no real remote attacker
    arrives from 127.0.0.0/8 or ::1.  ``safe_ips`` is operator-declared
    infrastructure (SSH gateway, monitoring, ...) that must stay reachable
    for the box to be manageable.  Matching is network-aware in both
    directions: a loopback CIDR (``127.0.0.0/8``) is refused too, and a
    source inside a safe CIDR is refused.

    Callers that mirror blocks into persistent layers (rule-engine
    blacklist + rules.json, the interceptor's ``_blocked`` bookkeeping)
    must apply the same test via ``IptablesManager.is_blockable()``,
    otherwise a refused block still lands in rules.json and survives
    restart as a phantom entry the kernel never enforced — and the API's
    two block views silently diverge.
    """
    try:
        net = ip_network(ip, strict=False)
    except ValueError:
        return False
    if net.is_loopback:
        return False
    for safe in safe_ips:
        try:
            safe_net = ip_network(safe, strict=False)
        except ValueError:
            continue
        # subnet_of raises TypeError across address families (e.g. an IPv4
        # source vs a "::1" safe entry) — those can never match.
        if safe_net.version != net.version:
            continue
        if net.subnet_of(safe_net):
            return False
    return True


class IptablesManager:
    """Manage iptables rules for traffic redirection and IP blocking.

    Every add_* method has a corresponding cleanup method.  Call
    cleanup_all() on graceful shutdown to restore the system to
    its pre-NIPS state.

    Requires root.

    Thread safety: block_ip() may be called from the detection event-loop
    thread while cleanup_all() runs on the main thread during teardown.
    All rule mutations — setup_nfqueue() included — are serialized by _lock.
    block_ip() additionally checks ``_nfqueue_rules_added`` *inside* the lock:
    if teardown has already deleted the chain, the block is skipped rather
    than raising CalledProcessError (which would otherwise be lost and desync
    the in-memory blocked set from the real firewall state).

    block_ip()/unblock_ip() return whether the kernel rule actually changed
    state.  Callers gate their own mirror layers on that result, so a refused
    or failed block never leaves a blacklist entry behind with nothing
    enforcing it.
    """

    CHAIN = "NIPS"

    def __init__(self, safe_ips: list[str] | None = None) -> None:
        self._safe_ips: list[str] = safe_ips or ["127.0.0.1"]
        self._blocked: set[str] = set()
        self._nfqueue_rules_added: bool = False
        # Number of ACCEPT guard rules (safe_ips, loopback, SSH) sitting at the
        # top of the chain after setup_nfqueue().  A block inserts its DROP at
        # _guard_rule_count + 1 — below the guards, above the NFQUEUE
        # redirects.  Inserting at position 1 put every DROP over the rules
        # whose whole purpose is to keep the box reachable.
        self._guard_rule_count: int = 0
        self._lock = threading.Lock()

    # --- nfqueue setup / teardown -----------------------------------------

    def setup_nfqueue(self, queue_num: int = 0, intercept_icmp: bool = False) -> None:
        """Redirect incoming TCP/UDP (and optionally ICMP) to NFQUEUE.

        Idempotent: safe to call when a previous run left the chain behind
        (e.g. after a crash).  ``_nfqueue_rules_added`` is set *before* any
        rule mutation so that ``cleanup_nfqueue`` always attempts teardown —
        even on a partial failure — preventing orphaned rules from
        desynchronizing the kernel firewall state.

        A chain that already exists is flushed first.  Its rules came from a
        run that did not shut down cleanly, so their provenance is unknown and
        ``_guard_rule_count`` could not be derived from them; rebuilding from
        empty is the only way to know where a later DROP has to go.  The flush
        leaves the chain briefly empty, which fails *open* for those few
        microseconds — acceptable because this runs once at startup, before
        capture begins, and the alternative is enforcing blocks at an unknown
        offset relative to rules nobody accounted for.
        """
        with self._lock:
            self._nfqueue_rules_added = True

            # ``-N`` fails (rc!=0) when the chain already exists.
            if self._rc("iptables", "-N", self.CHAIN) != 0:
                self._rc("iptables", "-F", self.CHAIN)

            if not self._rule_exists("INPUT", "-j", self.CHAIN):
                self._run("iptables", "-I", "INPUT", "-j", self.CHAIN)

            # Protect SSH and loopback.  Skip safe IPs that fail (e.g. IPv6 on
            # legacy iptables).
            guard_count = 0
            for ip in self._safe_ips:
                # IPv6 addresses belong in ip6tables; legacy `iptables -C/-I -s ::1`
                # behaves unpredictably (often errors rather than cleanly reporting
                # absence), so do NOT let _rule_exists' probe on ::1 masquerade as
                # "already installed" and silently skip the rule.  For IPv4 we still
                # probe to stay idempotent; for IPv6 we just attempt the insert and
                # tolerate failure.  Either way an ip6tables rule is not part of the
                # IPv4 chain, so it never counts toward the DROP offset.
                if ":" in ip:  # looks like IPv6
                    try:
                        self._run("ip6tables", "-I", self.CHAIN, "-s", ip, "-j", "ACCEPT")
                    except (subprocess.CalledProcessError, RuntimeError):
                        logger.warning("Could not add IPv6 safe IP %s — skipping", ip)
                    continue
                if self._insert_guard("-s", ip, "-j", "ACCEPT"):
                    guard_count += 1

            # Loopback never enters the pipeline.  Intercepting it caused 5s
            # detection timeouts on DNS replies from the 127.0.0.53 stub and,
            # in the worst case, a permanent self-DoS once the stub IP got
            # blocked.
            if self._insert_guard("-i", "lo", "-j", "ACCEPT"):
                guard_count += 1
            if self._insert_guard("-p", "tcp", "--dport", "22", "-j", "ACCEPT"):
                guard_count += 1
            self._guard_rule_count = guard_count

            # Redirect remaining TCP/UDP to NFQUEUE
            if not self._rule_exists(self.CHAIN, "-p", "tcp", "-j", "NFQUEUE", "--queue-num", str(queue_num)):
                self._run("iptables", "-A", self.CHAIN, "-p", "tcp",
                          "-j", "NFQUEUE", "--queue-num", str(queue_num))
            if not self._rule_exists(self.CHAIN, "-p", "udp", "-j", "NFQUEUE", "--queue-num", str(queue_num)):
                self._run("iptables", "-A", self.CHAIN, "-p", "udp",
                          "-j", "NFQUEUE", "--queue-num", str(queue_num))

            # ICMP is only inspected when explicitly asked for: the per-type
            # policy in RuleEngine can do anything useful once these packets
            # actually reach userspace.  Without this rule ICMP simply bypasses
            # the IPS rather than being blocked by it.
            if intercept_icmp and not self._rule_exists(
                    self.CHAIN, "-p", "icmp", "-j", "NFQUEUE", "--queue-num", str(queue_num)):
                self._run("iptables", "-A", self.CHAIN, "-p", "icmp",
                          "-j", "NFQUEUE", "--queue-num", str(queue_num))

        logger.info(
            "nfqueue rules added to iptables chain %s (%d guard rules)",
            self.CHAIN, self._guard_rule_count,
        )

    def _insert_guard(self, *spec: str) -> bool:
        """Ensure an ACCEPT guard is present in the chain.

        Returns True only when the rule really is there afterwards, so a
        failed insert does not inflate ``_guard_rule_count`` and push later
        DROPs one position too far down.
        """
        if self._rule_exists(self.CHAIN, *spec):
            return True
        try:
            self._run("iptables", "-I", self.CHAIN, *spec)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            logger.warning(
                "Could not add guard rule [%s] — skipping: %s", " ".join(spec), exc,
            )
            return False
        return True

    def cleanup_nfqueue(self) -> None:
        """Remove nfqueue rules. Safe to call even if not set up."""
        if not self._nfqueue_rules_added:
            return
        with self._lock:
            self._run("iptables", "-D", "INPUT", "-j", self.CHAIN, check=False)
            self._run("iptables", "-F", self.CHAIN, check=False)
            self._run("iptables", "-X", self.CHAIN, check=False)
            self._nfqueue_rules_added = False
            self._guard_rule_count = 0
        logger.info("nfqueue rules removed")

    # --- IP blocking -------------------------------------------------------

    def block_ip(self, ip: str) -> bool:
        """Install a kernel DROP for ``ip``.

        Returns True when the rule is in place afterwards — including the
        idempotent case where it already was — and False when the block was
        refused (loopback / safe-ip), skipped (chain gone) or rejected by
        iptables.  Callers must not commit any mirror layer on False: a
        blacklist entry with no kernel rule behind it is a phantom that
        survives restart and shows up in the API as a block nobody enforces.
        """
        with self._lock:
            if not blockable(ip, self._safe_ips):
                logger.warning("block_ip(%s) refused — loopback or safe-ip source", ip)
                return False
            # Teardown may have already deleted the chain on another thread.
            # Inserting into a non-existent chain raises CalledProcessError,
            # which would abort before updating ``_blocked`` and desync state
            # from the real firewall.  Skip the insert when the chain is gone.
            if not self._nfqueue_rules_added or not self._chain_exists(self.CHAIN):
                logger.warning(
                    "block_ip(%s) skipped — chain %s gone (likely during teardown)",
                    ip, self.CHAIN,
                )
                return False
            if ip in self._blocked:
                return True
            position = str(self._guard_rule_count + 1)
            try:
                # Below every ACCEPT guard (safe_ips, loopback, SSH) and above
                # the NFQUEUE redirects.
                self._run("iptables", "-I", self.CHAIN, position,
                          "-s", ip, "-j", "DROP")
            except (subprocess.CalledProcessError, RuntimeError):
                logger.warning("block_ip(%s) failed — iptables rejected the rule", ip)
                return False
            self._blocked.add(ip)
        logger.info("blocked IP: %s (chain position %s)", ip, position)
        return True

    def is_blockable(self, ip: str) -> bool:
        """True when block_ip(ip) would install (not refuse) a kernel DROP.

        The interceptor consults this before writing its mirror layers
        (rule-engine blacklist + rules.json, ``_blocked`` bookkeeping) so a
        refused block never becomes a phantom persistent entry — see
        blockable().
        """
        return blockable(ip, self._safe_ips)

    def unblock_ip(self, ip: str) -> bool:
        """Remove a kernel DROP.

        Returns False when there was nothing tracked to remove, or when the
        delete failed and the rule is still installed — the caller then keeps
        its own mirror in place, because dropping the mirror while the kernel
        still blocks would hide an enforced ban from every reporting layer.
        """
        with self._lock:
            if ip not in self._blocked:
                return False
            rc = self._rc("iptables", "-D", self.CHAIN, "-s", ip, "-j", "DROP")
            if rc != 0 and self._rule_exists(self.CHAIN, "-s", ip, "-j", "DROP"):
                logger.warning(
                    "unblock_ip(%s) failed — DROP rule still installed", ip,
                )
                return False
            self._blocked.discard(ip)
        logger.info("unblocked IP: %s", ip)
        return True

    def blocked_ips(self) -> list[str]:
        return sorted(self._blocked)

    # --- full cleanup ------------------------------------------------------

    def cleanup_all(self) -> None:
        # Lift the per-IP DROPs first: once the chain is deleted there is
        # nothing for unblock_ip to confirm against, so every call would
        # report success whether or not it had actually done anything.
        for ip in list(self._blocked):
            self.unblock_ip(ip)
        self.cleanup_nfqueue()

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _rc(*args) -> int:
        """Run an iptables command and return its exit code, never raising.

        For the call sites where a non-zero status is information rather than
        an error (``-N`` on an existing chain, ``-D`` on an absent rule).
        """
        try:
            return subprocess.run(
                list(args), capture_output=True, text=True, check=False,
            ).returncode
        except FileNotFoundError:
            return 127

    @staticmethod
    def _chain_exists(chain: str) -> bool:
        """Return True if the iptables chain exists.

        ``iptables -L <chain>`` succeeds (rc 0) exactly when the chain is
        present.  Do NOT use ``-C <chain>`` for this: ``-C`` checks a *rule
        specification*, and a bare chain name is "Bad rule" (rc 1) even when
        the chain exists — which silently disabled every block_ip() call.
        """
        try:
            result = subprocess.run(
                ["iptables", "-L", chain],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _rule_exists(*args) -> bool:
        """Return True if an iptables rule matching ``args`` already exists.

        ``iptables -C`` exits 0 when the rule is present and non-zero
        otherwise; ``check=False`` keeps it from raising.
        """
        try:
            result = subprocess.run(
                ["iptables", "-C", *args],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _run(*args, check: bool = True) -> str:
        try:
            result = subprocess.run(
                list(args),
                capture_output=True,
                text=True,
                check=check,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error("iptables error: %s", e.stderr.strip())
            raise
        except FileNotFoundError:
            raise RuntimeError("iptables not found — are you on Linux with root?")
