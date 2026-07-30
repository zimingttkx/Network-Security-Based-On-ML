"""Rule engine: fast IP/traffic filtering before ML analysis."""

from __future__ import annotations

import ipaddress
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Set

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict


class RateLimiter:
    """Sliding-window per-IP connection rate tracker."""

    def __init__(self, window_seconds: float = 1.0, max_connections: int = 100):
        self._window = window_seconds
        self._max_conn = max_connections
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str, timestamp: float) -> bool:
        """Return True if IP is within rate limit."""
        bucket = self._buckets[ip]
        cutoff = timestamp - self._window
        bucket[:] = [t for t in bucket if t > cutoff]
        bucket.append(timestamp)
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
        self._whitelist: Set[str] = set()
        self._blacklist: Set[str] = set()
        self._protocol_allow: Set[int] = {6, 17}  # TCP, UDP
        self._rate_limiter = RateLimiter()
        self._rules: list[dict] = []
        self._blocked_count: int = 0

    # -- public API ---------------------------------------------------------

    async def process_packet(self, packet: PacketInfo) -> Optional[Verdict]:
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
