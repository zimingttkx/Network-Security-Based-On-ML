"""iptables / nftables rule management (Linux only)."""

from __future__ import annotations

import logging
import subprocess
import threading
from ipaddress import ip_address

logger = logging.getLogger(__name__)


class IptablesManager:
    """Manage iptables rules for traffic redirection and IP blocking.

    Every add_* method has a corresponding cleanup method.  Call
    cleanup_all() on graceful shutdown to restore the system to
    its pre-NIPS state.

    Requires root.

    Thread safety: block_ip() may be called from the detection event-loop
    thread while cleanup_all() runs on the main thread during teardown.
    All rule mutations are serialized by _lock.  block_ip() additionally
    checks ``_nfqueue_rules_added`` *inside* the lock: if teardown has
    already deleted the chain, the block is skipped rather than raising
    CalledProcessError (which would otherwise be lost and desync the
    in-memory blocked set from the real firewall state).
    """

    CHAIN = "NIPS"

    def __init__(self, safe_ips: list[str] | None = None) -> None:
        self._safe_ips: list[str] = safe_ips or ["127.0.0.1"]
        self._blocked: set[str] = set()
        self._nfqueue_rules_added: bool = False
        self._lock = threading.Lock()

    # --- nfqueue setup / teardown -----------------------------------------

    def setup_nfqueue(self, queue_num: int = 0) -> None:
        """Redirect incoming TCP/UDP to NFQUEUE.

        Idempotent: safe to call when a previous run left the chain behind
        (e.g. after a crash).  ``_nfqueue_rules_added`` is set *before* any
        rule mutation so that ``cleanup_nfqueue`` always attempts teardown —
        even on a partial failure — preventing orphaned rules from
        desynchronizing the kernel firewall state.
        """
        self._nfqueue_rules_added = True

        # Create the chain if absent.  ``-N`` fails (rc!=0) when the chain
        # already exists; that is expected on a restart, so do not raise.
        self._run("iptables", "-N", self.CHAIN, check=False)

        if not self._rule_exists("INPUT", "-j", self.CHAIN):
            self._run("iptables", "-I", "INPUT", "-j", self.CHAIN)

        # Protect SSH and loopback.  Skip safe IPs that fail (e.g. IPv6 on legacy iptables).
        for ip in self._safe_ips:
            # IPv6 addresses belong in ip6tables; legacy `iptables -C/-I -s ::1`
            # behaves unpredictably (often errors rather than cleanly reporting
            # absence), so do NOT let _rule_exists' probe on ::1 masquerade as
            # "already installed" and silently skip the rule.  For IPv4 we still
            # probe to stay idempotent; for IPv6 we just attempt the insert and
            # tolerate failure (the caller's except swallows it as a warning).
            if ":" in ip:  # looks like IPv6
                try:
                    self._run("ip6tables", "-I", self.CHAIN, "-s", ip, "-j", "ACCEPT")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Could not add IPv6 safe IP %s — skipping", ip)
                continue
            if not self._rule_exists(self.CHAIN, "-s", ip, "-j", "ACCEPT"):
                try:
                    self._run("iptables", "-I", self.CHAIN, "-s", ip, "-j", "ACCEPT")
                except subprocess.CalledProcessError:
                    logger.warning("Could not add safe IP %s — skipping (likely unsupported on this system)", ip)

        # Loopback never enters the pipeline.  Intercepting it caused 5s
        # detection timeouts on DNS replies from the 127.0.0.53 stub and, in
        # the worst case, a permanent self-DoS once the stub IP got blocked.
        if not self._rule_exists(self.CHAIN, "-i", "lo", "-j", "ACCEPT"):
            self._run("iptables", "-A", self.CHAIN, "-i", "lo", "-j", "ACCEPT")

        if not self._rule_exists(self.CHAIN, "-p", "tcp", "--dport", "22", "-j", "ACCEPT"):
            self._run("iptables", "-A", self.CHAIN, "-p", "tcp", "--dport", "22",
                      "-j", "ACCEPT")

        # Redirect remaining TCP/UDP to NFQUEUE
        if not self._rule_exists(self.CHAIN, "-p", "tcp", "-j", "NFQUEUE", "--queue-num", str(queue_num)):
            self._run("iptables", "-A", self.CHAIN, "-p", "tcp",
                      "-j", "NFQUEUE", "--queue-num", str(queue_num))
        if not self._rule_exists(self.CHAIN, "-p", "udp", "-j", "NFQUEUE", "--queue-num", str(queue_num)):
            self._run("iptables", "-A", self.CHAIN, "-p", "udp",
                      "-j", "NFQUEUE", "--queue-num", str(queue_num))

        logger.info("nfqueue rules added to iptables chain %s", self.CHAIN)

    def cleanup_nfqueue(self) -> None:
        """Remove nfqueue rules. Safe to call even if not set up."""
        if not self._nfqueue_rules_added:
            return
        with self._lock:
            self._run("iptables", "-D", "INPUT", "-j", self.CHAIN, check=False)
            self._run("iptables", "-F", self.CHAIN, check=False)
            self._run("iptables", "-X", self.CHAIN, check=False)
            self._nfqueue_rules_added = False
        logger.info("nfqueue rules removed")

    # --- IP blocking -------------------------------------------------------

    def block_ip(self, ip: str) -> None:
        with self._lock:
            # Loopback sources are definitionally local traffic — this host,
            # its DNS stub resolver (127.0.0.53), or an internal service.
            # Blocking any of them is a self-DoS (observed live: the
            # systemd-resolved stub got a kernel DROP, silently killing host
            # DNS), and no real remote attacker arrives from 127.0.0.0/8 or ::1.
            try:
                loopback = ip_address(ip).is_loopback
            except ValueError:
                loopback = False
            if loopback:
                logger.warning("block_ip(%s) refused — loopback source", ip)
                return
            # Teardown may have already deleted the chain on another thread.
            # Inserting into a non-existent chain raises CalledProcessError,
            # which would abort before updating ``_blocked`` and desync state
            # from the real firewall.  Skip the insert when the chain is gone.
            if not self._nfqueue_rules_added or not self._chain_exists(self.CHAIN):
                logger.warning(
                    "block_ip(%s) skipped — chain %s gone (likely during teardown)",
                    ip, self.CHAIN,
                )
                return
            if ip in self._blocked or ip in self._safe_ips:
                return
            try:
                self._run("iptables", "-I", self.CHAIN, "1", "-s", ip, "-j", "DROP")
            except subprocess.CalledProcessError:
                logger.warning("block_ip(%s) failed — iptables rejected the rule", ip)
                return
            self._blocked.add(ip)
        logger.info("blocked IP: %s", ip)

    def unblock_ip(self, ip: str) -> None:
        with self._lock:
            if ip not in self._blocked:
                return
            self._run("iptables", "-D", self.CHAIN, "-s", ip, "-j", "DROP", check=False)
            self._blocked.discard(ip)
        logger.info("unblocked IP: %s", ip)

    def blocked_ips(self) -> list[str]:
        return sorted(self._blocked)

    # --- full cleanup ------------------------------------------------------

    def cleanup_all(self) -> None:
        self.cleanup_nfqueue()
        for ip in list(self._blocked):
            self.unblock_ip(ip)

    # --- helpers -----------------------------------------------------------

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
