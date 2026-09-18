"""Rule engine: fast IP/traffic filtering before ML analysis."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict

logger = logging.getLogger(__name__)

# A parsed CIDR rule (v4 or v6).
_Net = ipaddress.IPv4Network | ipaddress.IPv6Network


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
    2. Protocol   -> BLOCK (anything outside ``allowed_protocols``)
    3. Blacklist  -> BLOCK (persistent + ephemeral, see below)
    4. Rate limit -> BLOCK (new connections only: TCP SYN and UDP datagrams)
    5. None       -> pass to next detector

    Two blacklist tiers with different provenance:

    ``_blacklist`` holds operator entries and permanent bans.  It is what
    ``save_rules`` persists, so it survives a restart.

    ``_ephemeral_blacklist`` holds temp-ban mirrors installed by the block
    policy.  They have a TTL and are lifted by the expiry sweeper, so
    persisting them would resurrect expired bans after every restart — they
    are kept in a separate set that ``get_blacklist``/``save_rules`` never
    touch.  Matching checks the union.
    """

    def __init__(self, window_seconds: float = 1.0, max_connections: int = 1000,
                 allowed_protocols: set[int] | None = None) -> None:
        super().__init__(name="RuleEngine")
        self._whitelist: set[str] = set()
        self._blacklist: set[str] = set()
        self._ephemeral_blacklist: set[str] = set()
        # Pre-parsed CIDR entries.  Parsing every "/"-containing rule on every
        # packet made match cost linear in table size (ip_network() alone is
        # ~4us), so the lists are rebuilt on mutation and matching only does
        # address-in-network tests.
        self._wl_nets: list[_Net] = []
        self._bl_nets: list[_Net] = []
        self._bl_eph_nets: list[_Net] = []
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
        # Serializes save_rules on its own lock: a fixed "<name>.tmp" sidecar
        # made two concurrent savers race on the same path (one os.replace'd
        # the file the other was still writing -> FileNotFoundError), and
        # taking self._lock for the whole write would stall per-packet
        # matching behind disk I/O.
        self._save_lock = threading.Lock()

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

        # 3. Blacklist check (persistent + ephemeral temp-ban mirrors)
        if self._is_blacklisted(packet.src_ip):
            with self._lock:
                self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=1.0,
                           threat_level=ThreatLevel.HIGH,
                           reason="blacklist", detector=self.name)

        # 4. Rate limit — new connections only.  Counting every packet of an
        # established TCP session filled the bucket with a single bulk
        # transfer, so a legitimate peer got blocked mid-stream.
        if (self._counts_toward_rate(packet)
                and not self._rate_limiter.check(packet.src_ip, packet.timestamp)):
            with self._lock:
                self._blocked_count += 1
            return Verdict(action=Action.BLOCK, confidence=0.9,
                           threat_level=ThreatLevel.MEDIUM,
                           reason="rate limit exceeded", detector=self.name)

        return None  # pass

    @staticmethod
    def _counts_toward_rate(packet: PacketInfo) -> bool:
        """True for TCP SYN (ACK clear) and UDP datagrams.

        A "connection" for rate-limit purposes is a new session attempt:
        TCP handshake openers and connectionless UDP packets.  Pure ACKs,
        data segments and FIN/RST carry no new session and must not consume
        the budget.
        """
        if packet.protocol == 17:  # UDP
            return True
        if packet.protocol == 6:   # TCP: SYN set, ACK clear
            return packet.tcp_flags & 0x12 == 0x02
        return False

    # -- whitelist / blacklist management -----------------------------------

    def add_whitelist(self, entry: str) -> None:
        with self._lock:
            self._whitelist.add(entry)
            self._wl_nets = self._parse_nets(self._whitelist, "whitelist")

    def add_blacklist(self, entry: str) -> None:
        with self._lock:
            self._blacklist.add(entry)
            self._bl_nets = self._parse_nets(self._blacklist, "blacklist")

    def remove_whitelist(self, entry: str) -> bool:
        """Drop *entry*; return False when it was not present."""
        with self._lock:
            if entry not in self._whitelist:
                return False
            self._whitelist.discard(entry)
            self._wl_nets = self._parse_nets(self._whitelist, "whitelist")
            return True

    def remove_blacklist(self, entry: str) -> bool:
        """Drop a persistent *entry*; return False when it was not present."""
        with self._lock:
            if entry not in self._blacklist:
                return False
            self._blacklist.discard(entry)
            self._bl_nets = self._parse_nets(self._blacklist, "blacklist")
            return True

    def get_whitelist(self) -> list[str]:
        with self._lock:
            return sorted(self._whitelist)

    def get_blacklist(self) -> list[str]:
        """Persistent blacklist only — this is what gets saved to disk."""
        with self._lock:
            return sorted(self._blacklist)

    # -- ephemeral (temp-ban) blacklist -------------------------------------

    def add_ephemeral_blacklist(self, entry: str) -> None:
        """Mirror a temp ban into matching without making it persistent."""
        with self._lock:
            self._ephemeral_blacklist.add(entry)
            self._bl_eph_nets = self._parse_nets(self._ephemeral_blacklist,
                                                 "ephemeral blacklist")

    def remove_ephemeral_blacklist(self, entry: str) -> bool:
        """Lift a temp-ban mirror; return False when it was not present."""
        with self._lock:
            if entry not in self._ephemeral_blacklist:
                return False
            self._ephemeral_blacklist.discard(entry)
            self._bl_eph_nets = self._parse_nets(self._ephemeral_blacklist,
                                                 "ephemeral blacklist")
            return True

    def get_ephemeral_blacklist(self) -> list[str]:
        with self._lock:
            return sorted(self._ephemeral_blacklist)

    def promote_ephemeral(self, entry: str) -> bool:
        """Move *entry* from the ephemeral tier to the persistent one.

        Used when a temp ban escalates to permanent: the mirror stops being
        TTL-bound and becomes savable.  Returns False if there was no mirror
        to promote (the caller still gets a persistent entry).
        """
        with self._lock:
            found = entry in self._ephemeral_blacklist
            self._ephemeral_blacklist.discard(entry)
            self._bl_eph_nets = self._parse_nets(self._ephemeral_blacklist,
                                                 "ephemeral blacklist")
            self._blacklist.add(entry)
            self._bl_nets = self._parse_nets(self._blacklist, "blacklist")
            return found

    # -- persistence ---------------------------------------------------------

    def load_rules(self, path: Path) -> None:
        """Restore blacklist/whitelist from a JSON file.

        Only the persistent tiers are stored, so loading never resurrects an
        expired temp ban.
        """
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for ip in data.get("blacklist", []):
                self.add_blacklist(ip)
            for ip in data.get("whitelist", []):
                self.add_whitelist(ip)
        except Exception:
            logger.exception("Failed to load rules from %s", path)

    def save_rules(self, path: Path) -> None:
        """Persist the persistent blacklist/whitelist to a JSON file (atomic).

        Writes go to a uniquely named temp file in the destination directory,
        are flushed and fsync'd, then ``os.replace``'d into place.  A crash
        mid-write used to leave a truncated rules.json, and load_rules
        swallows parse errors — so every saved block (including permanent
        bans escalated by the block policy) would silently vanish on the next
        restart.  The unique name plus ``_save_lock`` also fixes concurrent
        savers racing on one shared ".tmp" path.
        """
        data = {
            "blacklist": self.get_blacklist(),
            "whitelist": self.get_whitelist(),
        }
        payload = json.dumps(data, indent=2)
        path = Path(path)
        with self._save_lock:
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                            prefix=path.name + ".",
                                            suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, path)
            except BaseException:
                # Never leave a stray sidecar behind on a failed save.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    # -- matching ------------------------------------------------------------

    def _is_whitelisted(self, ip: str) -> bool:
        with self._lock:
            if ip in self._whitelist:
                return True
            return self._in_nets(ip, self._wl_nets)

    def _is_blacklisted(self, ip: str) -> bool:
        with self._lock:
            if ip in self._blacklist or ip in self._ephemeral_blacklist:
                return True
            return (self._in_nets(ip, self._bl_nets)
                    or self._in_nets(ip, self._bl_eph_nets))

    @staticmethod
    def _in_nets(ip: str, nets: list[_Net]) -> bool:
        if not nets:
            return False
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        return any(addr in net for net in nets)

    @staticmethod
    def _parse_nets(entries: set[str], tier: str) -> list[_Net]:
        """Pre-parse the CIDR members of *entries*; skip and log invalid ones.

        ``strict=False``: operators write "10.0.0.5/24" meaning "the /24
        containing 10.0.0.5".  With the strict default that raises ValueError
        (host bits set) and the rule silently never matched anything.
        """
        nets: list[_Net] = []
        for entry in entries:
            if "/" not in entry:
                continue
            try:
                nets.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                logger.warning("ignoring invalid %s CIDR entry %r", tier, entry)
        return nets

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
                "ephemeral_blacklist_size": len(self._ephemeral_blacklist),
                "blocked_count": self._blocked_count,
                "packet_count": self._packet_count,
            }
