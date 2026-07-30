"""iptables / nftables rule management (Linux only)."""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class IptablesManager:
    """Manage iptables rules for traffic redirection and IP blocking.

    Every add_* method has a corresponding cleanup method.  Call
    cleanup_all() on graceful shutdown to restore the system to
    its pre-NIPS state.

    Requires root.
    """

    CHAIN = "NIPS"

    def __init__(self, safe_ips: Optional[list[str]] = None) -> None:
        self._safe_ips: list[str] = safe_ips or ["127.0.0.1"]
        self._blocked: set[str] = set()
        self._nfqueue_rules_added: bool = False

    # --- nfqueue setup / teardown -----------------------------------------

    def setup_nfqueue(self, queue_num: int = 0) -> None:
        """Redirect incoming TCP/UDP to NFQUEUE."""
        self._run("iptables", "-N", self.CHAIN)
        self._run("iptables", "-I", "INPUT", "-j", self.CHAIN)

        # Protect SSH and loopback.  Skip safe IPs that fail (e.g. IPv6 on legacy iptables).
        for ip in self._safe_ips:
            try:
                self._run("iptables", "-I", self.CHAIN, "-s", ip, "-j", "ACCEPT")
            except subprocess.CalledProcessError:
                logger.warning("Could not add safe IP %s — skipping (likely unsupported on this system)", ip)
        self._run("iptables", "-A", self.CHAIN, "-p", "tcp", "--dport", "22",
                  "-j", "ACCEPT")

        # Redirect remaining TCP/UDP to NFQUEUE
        self._run("iptables", "-A", self.CHAIN, "-p", "tcp",
                  "-j", "NFQUEUE", "--queue-num", str(queue_num))
        self._run("iptables", "-A", self.CHAIN, "-p", "udp",
                  "-j", "NFQUEUE", "--queue-num", str(queue_num))
        self._nfqueue_rules_added = True
        logger.info("nfqueue rules added to iptables chain %s", self.CHAIN)

    def cleanup_nfqueue(self) -> None:
        """Remove nfqueue rules. Safe to call even if not set up."""
        if not self._nfqueue_rules_added:
            return
        self._run("iptables", "-D", "INPUT", "-j", self.CHAIN, check=False)
        self._run("iptables", "-F", self.CHAIN, check=False)
        self._run("iptables", "-X", self.CHAIN, check=False)
        self._nfqueue_rules_added = False
        logger.info("nfqueue rules removed")

    # --- IP blocking -------------------------------------------------------

    def block_ip(self, ip: str) -> None:
        if ip in self._blocked or ip in self._safe_ips:
            return
        self._run("iptables", "-I", self.CHAIN, "1", "-s", ip, "-j", "DROP")
        self._blocked.add(ip)
        logger.info("blocked IP: %s", ip)

    def unblock_ip(self, ip: str) -> None:
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
