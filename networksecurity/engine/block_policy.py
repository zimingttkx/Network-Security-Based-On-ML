"""Escalation policy for BLOCK verdicts: strike counting + temp/perm bans.

Design (decided 2026-09): a BLOCK verdict is an *event*, not a sentence.
A single ML anomaly must never install a permanent kernel DROP — the
anomaly detector carries an irreducible false-positive rate (measured
8-13% on long-run normal traffic after the wall-clock feature fix), and
the old one-strike path burned legitimate sources into rules.json with
no recovery path.

Three escalating states per source IP:

    observing   not yet blocked; strikes accumulate inside a rolling window
    temp_banned kernel DROP + rule-engine blacklist entry with a TTL;
                lifted automatically by the expiry sweeper (strikes kept)
    perm_banned permanent DROP, mirrored into the persisted blacklist;
                only an explicit operator unblock lifts it

All knobs come from config.yaml's ``blocking:`` block (see
``load_blocking_config``) so operators can tune the policy without
touching code.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Literal

logger = logging.getLogger(__name__)

BlockState = Literal["observing", "temp_banned", "perm_banned"]


@dataclass
class BlockRecord:
    """Escalation state for one source IP."""

    ip: str
    # Timestamps of the BLOCKs inside the rolling window.  The previous
    # ``strikes`` counter plus a ``window_start`` was a tumbling window: once
    # a BLOCK arrived more than ``strikes_window`` after ``window_start`` the
    # whole counter was wiped, discarding strikes that were still recent
    # evidence.  Measured with threshold=3/window=300 and BLOCKs at
    # t=0,299,301,302: tumbling reset at t=301 and only reached 2 strikes by
    # t=302, so the configured escalation never fired even though three BLOCKs
    # landed inside 300s.  Keeping the timestamps lets old strikes fall off
    # one at a time.
    strike_times: deque[float] = field(default_factory=deque)
    state: BlockState = "observing"
    temp_ban_until: float | None = None
    temp_ban_count: int = 0          # completed temp bans -> perm escalation
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def strikes(self) -> int:
        """BLOCKs currently inside the rolling window."""
        return len(self.strike_times)

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "strikes": len(self.strike_times),
            "state": self.state,
            "temp_ban_until": self.temp_ban_until,
            "temp_ban_count": self.temp_ban_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class BlockPolicy:
    """Bounded LRU table mapping source IP -> escalation state.

    Pure policy: it decides *whether* and *how* to block, and delegates the
    actual enforcement through the callables given at construction so it
    stays unit-testable without iptables.
    """

    def __init__(
        self,
        strikes_threshold: int = 5,
        strikes_window: float = 300.0,
        temp_ban_seconds: float = 600.0,
        temp_ban_count_to_perm: int = 3,
        table_max: int = 50_000,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.strikes_threshold = max(1, strikes_threshold)
        self.strikes_window = max(1.0, strikes_window)
        self.temp_ban_seconds = max(1.0, temp_ban_seconds)
        self.temp_ban_count_to_perm = max(1, temp_ban_count_to_perm)
        self.table_max = max(1, table_max)
        self._now = now
        self._records: "OrderedDict[str, BlockRecord]" = OrderedDict()
        self._lock = threading.Lock()

    # -- core decision -------------------------------------------------------

    def record_block(self, ip: str) -> tuple[bool, BlockRecord]:
        """Account one BLOCK verdict for ``ip``.

        Returns ``(should_enforce, record)``.  ``should_enforce`` is True
        only when this call *escalates* the IP into a ban state — the caller
        installs enforcement exactly once per escalation.  Escalation to
        ``perm_banned`` happens inside the temp ban (the source kept
        offending after ``temp_ban_count_to_perm`` completed bans), so the
        caller must check ``record.state`` to know which enforcement to
        install.
        """
        now = self._now()
        with self._lock:
            rec = self._records.get(ip)
            if rec is None:
                rec = BlockRecord(ip=ip, first_seen=now)
                self._insert(rec)
            rec.last_seen = now
            self._records.move_to_end(ip)

            # Sliding window: strikes older than the window are stale evidence
            # and fall off individually, so a source with one anomaly a day
            # never accumulates towards a ban.
            cutoff = now - self.strikes_window
            while rec.strike_times and rec.strike_times[0] <= cutoff:
                rec.strike_times.popleft()
            rec.strike_times.append(now)

            if rec.state == "perm_banned":
                return False, rec

            if rec.state == "temp_banned":
                # Still banned and still offending — extend the TTL.  If this
                # source has already served temp_ban_count_to_perm bans, stop
                # rotating: escalate to permanent right here.
                rec.temp_ban_until = now + self.temp_ban_seconds
                if rec.temp_ban_count >= self.temp_ban_count_to_perm:
                    rec.state = "perm_banned"
                    rec.temp_ban_until = None
                    return True, rec
                return False, rec

            if len(rec.strike_times) >= self.strikes_threshold:
                if rec.temp_ban_count >= self.temp_ban_count_to_perm:
                    # Served the full temp-ban quota already and it is back
                    # on the threshold: no more rotation, permanent ban.
                    rec.state = "perm_banned"
                    rec.temp_ban_until = None
                    return True, rec
                rec.temp_ban_count += 1
                rec.state = "temp_banned"
                rec.temp_ban_until = now + self.temp_ban_seconds
                return True, rec

            return False, rec

    def expire_temp_bans(self) -> list[str]:
        """Lift every temp ban whose TTL has passed.

        Returns the list of IPs that left temp_banned this call.  Strikes are
        reset (each ban cycle must re-earn the threshold on its own) but the
        completed-ban counter is kept — that counter is what eventually
        escalates a repeat offender to perm_banned (checked in record_block
        while the source is still offending inside a ban, or on the strike
        that would start a new ban past the count).
        """
        now = self._now()
        lifted: list[str] = []
        with self._lock:
            for rec in self._records.values():
                if (rec.state == "temp_banned"
                        and rec.temp_ban_until is not None
                        and rec.temp_ban_until <= now):
                    rec.state = "observing"
                    rec.temp_ban_until = None
                    rec.strike_times.clear()
                    lifted.append(rec.ip)
        return lifted

    def unblock(self, ip: str) -> bool:
        """Operator-initiated full unblock (drops the record entirely)."""
        with self._lock:
            rec = self._records.pop(ip, None)
        return rec is not None

    def get(self, ip: str) -> BlockRecord | None:
        with self._lock:
            return self._records.get(ip)

    def snapshot(self) -> list[dict]:
        """All records for the /api/v1/blocks endpoint, most recent first."""
        with self._lock:
            recs = sorted(self._records.values(),
                          key=lambda r: r.last_seen, reverse=True)
            return [r.to_dict() for r in recs]

    # -- internals -----------------------------------------------------------

    def _insert(self, rec: BlockRecord) -> None:
        """Insert a record, evicting an LRU *observing* entry over capacity.

        Only observing records are safe to evict.  A temp_banned/perm_banned
        record is the sole holder of the ban's TTL and escalation count: drop
        it and the kernel DROP plus blacklist mirror stay installed forever
        with nothing left to lift them.  The candidate scan is O(n) but only
        runs at capacity.  If the table is entirely active bans there is no
        safe victim, so the oldest active record is dropped with a WARNING
        naming the IP whose enforcement now needs a manual lift.
        """
        if len(self._records) >= self.table_max:
            victim = next((ip for ip, r in self._records.items()
                           if r is not rec and r.state == "observing"), None)
            if victim is None:
                victim = next((ip for ip, r in self._records.items()
                               if r is not rec), None)
                if victim is not None:
                    logger.warning(
                        "block table full (%d records, all carrying live bans); "
                        "evicting %s (state=%s) — its enforcement must be lifted "
                        "manually", len(self._records), victim,
                        self._records[victim].state,
                    )
            if victim is not None:
                del self._records[victim]
        self._records[rec.ip] = rec


# Re-exported for typing convenience in Interceptor/API layers.
__all__ = ["BlockPolicy", "BlockRecord", "BlockState"]
