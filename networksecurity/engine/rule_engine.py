"""Rule engine: fast IP/traffic filtering before ML analysis."""

from __future__ import annotations

import ipaddress
import json
import logging
from collections import defaultdict
from pathlib import Path

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict


class RateLimiter:
    """Sliding-window per-IP connection rate tracker.

    Buckets are evicted once they fall fully outside the window and the
    tracked-IP set grows past ``max_buckets`` (LRU), so memory stays
    bounded under long-running live interception.
    """

    def __init__(self, window_seconds: float = 1.0, max_connections: int = 100,
                 max_buckets: int = 100000):
        self._window = window_seconds
        self._max_conn = max_connections
        self._max_buckets = max(1, max_buckets)
        self._buckets: dict[str, list[float]] = {}

    def check(self, ip: str, timestamp: float) -> bool:
        """Return True if IP is within rate limit."""
        bucket = self._buckets.get(ip)
        cutoff = timestamp - self._window
        if bucket is None:
            bucket = []
            self._buckets[ip] = bucket
        bucket[:] = [t for t in bucket if t > cutoff]
        bucket.append(timestamp)
        # Evict fully-expired buckets and cap total size (LRU-ish: drop first).
        if len(self._buckets) > self._max_buckets:
            expired = [k for k, v in self._buckets.items()
                       if not any(t > cutoff for t in v)]
            for k in expired[:len(self._buckets) - self._max_buckets]:
                self._buckets.pop(k, None)
            # If still over cap (all active), drop the oldest tracked key.
            while len(self._buckets) > self._max_buckets:
                self._buckets.pop(next(iter(self._buckets)), None)
        return len(bucket) <= self._max_conn

    def reset(self, ip: str = "") -> None:
        if ip:
            self._buckets.pop(ip, None)
        else:
            self._buckets.clear()


class RuleEngine(BaseDetector):
    """Multi-stage pre-filter applied before ML-based detectors.

    Stages (short-circuit on first match):
    1. Whitelist  -> ALLOW
    2. Blacklist  -> BLOCK
    3. Rate limit -> BLOCK
    4. None       -> pass to next detector
    """

    def __init__(self) -> None:
        super().__init__(name="RuleEngine")
        self._whitelist: set[str] = set()
        self._blacklist: set[str] = set()
        # Protocols allowed through.  TCP(6) and UDP(17) are the data
        # carriers; ICMP(1) is permitted by default so pings, PMTU and
        # error messages keep working.  ARP/other non-IP are handled
        # elsewhere.  Uncomment to tighten: {6, 17}.
        self._protocol_allow: set[int] = {1, 6, 17}  # ICMP, TCP, UDP
        self._rate_limiter = RateLimiter()
        self._rules: list[dict] = []
        self._blocked_count: int = 0

    # -- public API ---------------------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        self._packet_count += 1

        # 1. Whitelist check
        if self._is_whitelisted(packet.src_ip):
            return Verdict(action=Action.ALLOW, confidence=1.0,
                           threat_level=ThreatLevel.SAFE,
                           reason="whitelist", detector=self.name)

        # 2. Protocol filter
        if packet.protocol not in self._protocol_allow:
            self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=1.0,
                           threat_level=ThreatLevel.MEDIUM,
                           reason=f"protocol {packet.protocol} not allowed",
                           detector=self.name)

        # 3. Blacklist check
        if self._is_blacklisted(packet.src_ip):
            self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=1.0,
                           threat_level=ThreatLevel.HIGH,
                           reason="blacklist", detector=self.name)

        # 4. Rate limit
        if not self._rate_limiter.check(packet.src_ip, packet.timestamp):
            self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=0.9,
                           threat_level=ThreatLevel.MEDIUM,
                           reason="rate limit exceeded", detector=self.name)

        return None  # pass

    # -- whitelist / blacklist management -----------------------------------

    def add_whitelist(self, entry: str) -> None:
        self._whitelist.add(entry)

    def add_blacklist(self, entry: str) -> None:
        self._blacklist.add(entry)

    def remove_whitelist(self, entry: str) -> None:
        self._whitelist.discard(entry)

    def remove_blacklist(self, entry: str) -> None:
        self._blacklist.discard(entry)

    def get_whitelist(self) -> list[str]:
        return sorted(self._whitelist)

    def get_blacklist(self) -> list[str]:
        return sorted(self._blacklist)

    # -- persistence ---------------------------------------------------------

    def load_rules(self, path: Path) -> None:
        """Restore blacklist/whitelist from a JSON file."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for ip in data.get("blacklist", []):
                self.add_blacklist(ip)
            for ip in data.get("whitelist", []):
                self.add_whitelist(ip)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Failed to load rules from %s", path)

    def save_rules(self, path: Path) -> None:
        """Persist blacklist/whitelist to a JSON file."""
        data = {
            "blacklist": self.get_blacklist(),
            "whitelist": self.get_whitelist(),
        }
        path.write_text(json.dumps(data, indent=2))

    def _is_whitelisted(self, ip: str) -> bool:
        return ip in self._whitelist or any(
            self._ip_in_network(ip, entry)
            for entry in self._whitelist if "/" in entry
        )

    def _is_blacklisted(self, ip: str) -> bool:
        return ip in self._blacklist or any(
            self._ip_in_network(ip, entry)
            for entry in self._blacklist if "/" in entry
        )

    @staticmethod
    def _ip_in_network(ip: str, network: str) -> bool:
        try:
            return ipaddress.ip_address(ip) in ipaddress.ip_network(network)
        except ValueError:
            return False

    @property
    def blocked_count(self) -> int:
        return self._blocked_count

    def reset(self) -> None:
        super().reset()
        self._blocked_count = 0
        self._rate_limiter.reset()

    def stats(self) -> dict:
        return {
            "whitelist_size": len(self._whitelist),
            "blacklist_size": len(self._blacklist),
            "blocked_count": self._blocked_count,
            "packet_count": self._packet_count,
        }
