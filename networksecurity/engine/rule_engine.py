"""Rule engine: fast IP/traffic filtering before ML analysis."""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict


class RateLimiter:
    """Sliding-window per-IP connection rate tracker.

    ``_buckets`` is an OrderedDict used as an LRU: every touch refreshes the
    key's position, and once the tracked-IP set grows past ``max_buckets``
    the least-recently-seen bucket is evicted with ``popitem(last=False)``
    (O(1)).  Eviction is deliberately heuristic — a bucket not touched
    within the window is dead anyway, so scanning the whole table to find
    "expired" buckets first (the previous O(n)-per-packet behavior, trivially
    weaponizable with 100k+ spoofed sources) buys nothing over plain LRU.
    """

    def __init__(self, window_seconds: float = 1.0, max_connections: int = 100,
                 max_buckets: int = 100000):
        self._window = window_seconds
        self._max_conn = max_connections
        self._max_buckets = max(1, max_buckets)
        self._buckets: "OrderedDict[str, list[float]]" = OrderedDict()

    def check(self, ip: str, timestamp: float) -> bool:
        """Return True if IP is within rate limit (under the connection cap).

        A non-positive ``timestamp`` (e.g. the ``0.0`` default when a packet
        carries no timing information) cannot form a valid sliding window, so
        the check is skipped for that IP instead of counting it — otherwise
        the bucket would never expire and the IP would be blocked permanently
        after ``max_connections`` such packets.
        """
        if timestamp <= 0:
            return True
        bucket = self._buckets.get(ip)
        cutoff = timestamp - self._window
        if bucket is None:
            bucket = []
            self._buckets[ip] = bucket
        else:
            self._buckets.move_to_end(ip)  # LRU refresh
        # Drop entries that fell outside the sliding window.
        bucket[:] = [t for t in bucket if t > cutoff]
        bucket.append(timestamp)
        # Cap total size: O(1) eviction of the least-recently-seen bucket.
        while len(self._buckets) > self._max_buckets:
            self._buckets.popitem(last=False)
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

    def __init__(self, window_seconds: float = 1.0, max_connections: int = 1000,
                 allowed_protocols: set[int] | None = None) -> None:
        super().__init__(name="RuleEngine")
        self._whitelist: set[str] = set()
        self._blacklist: set[str] = set()
        # Protocols allowed through.  Inline IPS: only TCP(6) and UDP(17)
        # are passed; everything else (ICMP, etc.) is blocked by default.
        # Overridable via config.yaml -> engine.rule_engine.allowed_protocols.
        self._protocol_allow: set[int] = set(
            allowed_protocols) if allowed_protocols else {6, 17}
        # Sliding-window rate limiter.  The code default cap is generous
        # (1000 conns/s per source IP) to avoid false-blocking busy-but-
        # legitimate clients; config.yaml -> engine.rule_engine.rate_limit
        # overrides it when present (shipped config: 100).
        self._rate_limiter = RateLimiter(window_seconds=window_seconds,
                                         max_connections=max_connections)
        self._rules: list[dict] = []
        self._blocked_count: int = 0
        # Guards all whitelist/blacklist mutations and reads.  Rules are
        # edited from the API thread (rules CRUD endpoints) while being read
        # on every packet by the detection loop thread; without this lock a
        # concurrent edit can raise "Set changed size during iteration" inside
        # _is_blacklisted/_is_whitelisted and force a fail-closed drop of all
        # traffic.
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        # Counters are mutated under self._lock so the API thread reading
        # stats() sees a consistent snapshot (no lost updates under
        # concurrent packet processing).
        with self._lock:
            self._packet_count += 1

        # 1. Whitelist check
        if self._is_whitelisted(packet.src_ip):
            return Verdict(action=Action.ALLOW, confidence=1.0,
                           threat_level=ThreatLevel.SAFE,
                           reason="whitelist", detector=self.name)

        # 2. Protocol filter
        if packet.protocol not in self._protocol_allow:
            with self._lock:
                self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=1.0,
                           threat_level=ThreatLevel.MEDIUM,
                           reason=f"protocol {packet.protocol} not allowed",
                           detector=self.name)

        # 3. Blacklist check
        if self._is_blacklisted(packet.src_ip):
            with self._lock:
                self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=1.0,
                           threat_level=ThreatLevel.HIGH,
                           reason="blacklist", detector=self.name)

        # 4. Rate limit
        if not self._rate_limiter.check(packet.src_ip, packet.timestamp):
            with self._lock:
                self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=0.9,
                           threat_level=ThreatLevel.MEDIUM,
                           reason="rate limit exceeded", detector=self.name)

        return None  # pass

    # -- whitelist / blacklist management -----------------------------------

    def add_whitelist(self, entry: str) -> None:
        with self._lock:
            self._whitelist.add(entry)

    def add_blacklist(self, entry: str) -> None:
        with self._lock:
            self._blacklist.add(entry)

    def remove_whitelist(self, entry: str) -> None:
        with self._lock:
            self._whitelist.discard(entry)

    def remove_blacklist(self, entry: str) -> None:
        with self._lock:
            self._blacklist.discard(entry)

    def get_whitelist(self) -> list[str]:
        with self._lock:
            return sorted(self._whitelist)

    def get_blacklist(self) -> list[str]:
        with self._lock:
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
        with self._lock:
            return ip in self._whitelist or any(
                self._ip_in_network(ip, entry)
                for entry in self._whitelist if "/" in entry
            )

    def _is_blacklisted(self, ip: str) -> bool:
        with self._lock:
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
        with self._lock:
            return {
                "whitelist_size": len(self._whitelist),
                "blacklist_size": len(self._blacklist),
                "blocked_count": self._blocked_count,
                "packet_count": self._packet_count,
            }
