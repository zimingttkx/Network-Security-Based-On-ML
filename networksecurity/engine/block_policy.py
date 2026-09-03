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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Literal

logger = logging.getLogger(__name__)

BlockState = Literal["observing", "temp_banned", "perm_banned"]


@dataclass
class BlockRecord:
    """Escalation state for one source IP."""

    ip: str
    strikes: int = 0                 # BLOCKs inside the current window
    window_start: float = 0.0        # timestamp the strike window opened
    state: BlockState = "observing"
    temp_ban_until: float | None = None
    temp_ban_count: int = 0          # completed temp bans -> perm escalation
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "strikes": self.strikes,
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
                rec = BlockRecord(ip=ip, first_seen=now, window_start=now)
                self._insert(rec)
            rec.last_seen = now
            self._records.move_to_end(ip)

            # Strike window semantics: a counter older than the window means
            # stale evidence, so the counter restarts rather than compounding
            # (a source with one anomaly a day never escalates).
            if now - rec.window_start > self.strikes_window:
                rec.window_start = now
                rec.strikes = 0

            rec.strikes += 1

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

            if rec.strikes >= self.strikes_threshold:
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
                    rec.window_start = now
                    rec.strikes = 0
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
        """Insert a record, evicting the LRU entry over capacity.

        Eviction only drops *observing* records' eligibility; an evicted
        temp_banned/perm_banned IP's enforcement (iptables + blacklist)
        lives outside this table and is unaffected.  The new record itself
        is never an eviction candidate (table_max >= 1 guarantees room once
        the evictions run before the insert).
        """
        while len(self._records) >= self.table_max:
            _, evicted = self._records.popitem(last=False)
            if evicted is not rec and evicted.state != "observing":
                logger.debug(
                    "evicted active record %s (state=%s) — enforcement persists",
                    evicted.ip, evicted.state,
                )
        self._records[rec.ip] = rec


# Re-exported for typing convenience in Interceptor/API layers.
__all__ = ["BlockPolicy", "BlockRecord", "BlockState"]
