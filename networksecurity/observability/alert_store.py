"""Durable event storage for NIPS: security alerts and the management audit trail.

Both streams land in one SQLite database (stdlib sqlite3, no new dependency)
and share one writer thread.  Two properties drive the design:

1. The packet path must never block on the disk.  ``record_alert`` is called
   from the NFQUEUE/detection path, where a stall turns into a fail-closed
   drop of every in-flight packet.  Producers therefore only enqueue into a
   bounded queue; one background thread performs batched INSERTs inside a
   single transaction per batch.  When the queue is full, events are dropped
   and counted rather than applying back-pressure to detection.

2. A security event log that loses records silently is worse than none, so
   every drop, write failure and prune is counted and exposed via ``stats()``
   instead of only logged, and the most recent events are mirrored in an
   in-memory ring so reads still answer if the database is unusable.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Table -> columns in INSERT order.  "ts" is epoch seconds; readers get ISO8601.
_ALERT_SQL = "INSERT INTO alerts (ts, source_ip, reason, action, detector) VALUES (?,?,?,?,?)"
_AUDIT_SQL = ("INSERT INTO audit (ts, actor, method, path, target, result, detail)"
              " VALUES (?,?,?,?,?,?,?)")
_ALERT_KEYS = ("source_ip", "reason", "action", "detector")
_AUDIT_KEYS = ("actor", "method", "path", "target", "result", "detail")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY,
    ts        REAL NOT NULL,
    source_ip TEXT NOT NULL,
    reason    TEXT NOT NULL,
    action    TEXT NOT NULL,
    detector  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_ts  ON alerts (ts);
CREATE INDEX IF NOT EXISTS alerts_src ON alerts (source_ip);

CREATE TABLE IF NOT EXISTS audit (
    id     INTEGER PRIMARY KEY,
    ts     REAL NOT NULL,
    actor  TEXT NOT NULL,
    method TEXT NOT NULL,
    path   TEXT NOT NULL,
    target TEXT NOT NULL,
    result TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_ts ON audit (ts);
"""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _tail(ring: deque, limit: int, offset: int) -> list[dict]:
    """Page a newest-first in-memory ring (the degraded read path)."""
    items = list(ring)[::-1]
    return items[offset:offset + max(1, min(int(limit), 1000))]


class EventStore:
    """Append-mostly SQLite store for alert and audit records."""

    def __init__(self, path: str | Path, *, max_rows: int = 200_000,
                 retention_days: float = 30.0, queue_size: int = 10_000,
                 batch_size: int = 200, flush_interval: float = 0.25) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.retention_days = retention_days
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.dropped = 0                 # queue full: producer never blocked
        self.write_errors = 0            # batches that failed to land or read
        self.written = 0
        self.purged = 0

        self._queue: deque[tuple[str, dict]] = deque()
        self._qmax = queue_size
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._lock = threading.Lock()    # serialises access to the connection
        self._recent_alerts: deque[dict] = deque(maxlen=500)
        self._recent_audit: deque[dict] = deque(maxlen=500)
        self._closed = False

        self._conn: sqlite3.Connection | None = None
        self._open_failed = False
        self._open()

        self._thread = threading.Thread(target=self._writer_loop, daemon=True,
                                        name="nips-event-store")
        self._thread.start()

    # -- connection ----------------------------------------------------------

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
        except (sqlite3.Error, OSError, ValueError) as exc:
            # ValueError is in the list because sqlite raises it (not
            # sqlite3.Error) for an unusable path such as an embedded NUL, and
            # a bad storage.events_db must degrade to the ring buffer rather
            # than abort process startup.
            logger.error("event store %s unusable (%s); serving reads from the "
                         "in-memory ring only", self.path, exc)
            self._open_failed = True
            self._conn = None
            return
        self._conn = conn

    @property
    def degraded(self) -> bool:
        return self._conn is None

    # -- producers -----------------------------------------------------------

    def record_alert(self, source_ip: str, reason: str, action: str, detector: str,
                     *, ts: float | None = None) -> None:
        """Queue one detection/block event.  Never blocks the caller."""
        self._enqueue("alert", {"ts": ts if ts is not None else time.time(),
                                "source_ip": source_ip, "reason": reason,
                                "action": action, "detector": detector})

    def record_audit(self, actor: str, method: str, path: str, target: str,
                     result: str, detail: str = "", *,
                     ts: float | None = None) -> None:
        """Queue one management-plane action for the audit trail."""
        self._enqueue("audit", {"ts": ts if ts is not None else time.time(),
                                "actor": actor, "method": method, "path": path,
                                "target": target, "result": result, "detail": detail})

    def _enqueue(self, kind: str, row: dict) -> None:
        ts = row["ts"]
        view = {"timestamp": _iso(ts)}
        view.update({k: row[k] for k in (_ALERT_KEYS if kind == "alert" else _AUDIT_KEYS)})
        (self._recent_alerts if kind == "alert" else self._recent_audit).append(view)
        with self._cond:
            if len(self._queue) >= self._qmax:
                self.dropped += 1
                if self.dropped in (1, 10, 100) or self.dropped % 1000 == 0:
                    logger.warning("event store queue full; %d events dropped so far",
                                   self.dropped)
                return
            self._queue.append((kind, row))
            self._cond.notify()

    # -- writer thread -------------------------------------------------------

    def _writer_loop(self) -> None:
        while not self._stop.is_set():
            batch = self._drain_batch()
            if batch:
                self._write(batch)
        with self._cond:                 # final drain on close(): pop as we take
            batch = [self._queue.popleft() for _ in range(len(self._queue))]
        self._write(batch)

    def _drain_batch(self) -> list[tuple[str, dict]]:
        """Wait up to flush_interval for work, then take at most batch_size items."""
        deadline = time.monotonic() + self.flush_interval
        with self._cond:
            while not self._queue and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cond.wait(remaining)
            take = min(self.batch_size, len(self._queue))
            return [self._queue.popleft() for _ in range(take)]

    def _write(self, batch: list[tuple[str, dict]]) -> None:
        if not batch or self._conn is None:
            return
        alerts = [r for kind, r in batch if kind == "alert"]
        audit = [r for kind, r in batch if kind == "audit"]
        try:
            with self._lock:
                cur = self._conn.cursor()
                if alerts:
                    cur.executemany(_ALERT_SQL,
                                    [[r["ts"]] + [r[k] for k in _ALERT_KEYS] for r in alerts])
                if audit:
                    cur.executemany(_AUDIT_SQL,
                                    [[r["ts"]] + [r[k] for k in _AUDIT_KEYS] for r in audit])
                self.written += len(batch)
                self._prune(cur)
                self._conn.commit()
        except sqlite3.Error as exc:
            self.write_errors += 1
            logger.error("event store batch failed (%s); %d events lost", exc, len(batch))

    def _prune(self, cur: sqlite3.Cursor) -> None:
        """Bound by age and by size.

        Age alone is not enough: a busy host writes millions of rows inside one
        retention window, and the file would grow without limit.
        """
        cutoff = time.time() - self.retention_days * 86400.0
        for table in ("alerts", "audit"):
            spec = self._SQL[table]
            cur.execute(spec["purge_age"], (cutoff,))
            if cur.rowcount > 0:
                self.purged += cur.rowcount
            cur.execute(spec["count_all"])
            total = cur.fetchone()[0]
            if total > self.max_rows:
                cur.execute(spec["purge_size"], (total - self.max_rows,))
                if cur.rowcount > 0:
                    self.purged += cur.rowcount

    # -- readers -------------------------------------------------------------

    def query_alerts(self, *, limit: int = 100, offset: int = 0, source_ip: str | None = None,
                     action: str | None = None, since: float | None = None,
                     until: float | None = None) -> dict:
        return self._query("alerts", (source_ip, action),
                           limit=limit, offset=offset, since=since, until=until)

    def query_audit(self, *, limit: int = 100, offset: int = 0, actor: str | None = None,
                    result: str | None = None, since: float | None = None,
                    until: float | None = None) -> dict:
        return self._query("audit", (actor, result),
                           limit=limit, offset=offset, since=since, until=until)

    # Every statement is a literal: no caller-supplied text is ever concatenated
    # into SQL, and each value arrives as a bound parameter.  Optional equality
    # filters use "(? IS NULL OR col = ?)" so a single statement covers every
    # combination.  Time bounds use finite sentinels rather than float('inf') —
    # SQLite converts IEEE infinities to NULL, which would silently drop rows.
    _SQL = {
        "alerts": {
            "keys": _ALERT_KEYS,
            "count": ("SELECT count(*) FROM alerts WHERE ts >= ? AND ts <= ?"
                      " AND (? IS NULL OR source_ip = ?)"
                      " AND (? IS NULL OR action = ?)"),
            "page": ("SELECT ts, source_ip, reason, action, detector FROM alerts"
                     " WHERE ts >= ? AND ts <= ?"
                     " AND (? IS NULL OR source_ip = ?)"
                     " AND (? IS NULL OR action = ?)"
                     " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?"),
            "count_all": "SELECT count(*) FROM alerts",
            "purge_age": "DELETE FROM alerts WHERE ts < ?",
            # Order by ts, not id: a replayed/backfilled event can carry an
            # older timestamp than its row id, and the cap must shed the
            # oldest events.
            "purge_size": "DELETE FROM alerts WHERE id IN"
                          " (SELECT id FROM alerts ORDER BY ts ASC, id ASC LIMIT ?)",
        },
        "audit": {
            "keys": _AUDIT_KEYS,
            "count": ("SELECT count(*) FROM audit WHERE ts >= ? AND ts <= ?"
                      " AND (? IS NULL OR actor = ?)"
                      " AND (? IS NULL OR result = ?)"),
            "page": ("SELECT ts, actor, method, path, target, result, detail FROM audit"
                     " WHERE ts >= ? AND ts <= ?"
                     " AND (? IS NULL OR actor = ?)"
                     " AND (? IS NULL OR result = ?)"
                     " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?"),
            "count_all": "SELECT count(*) FROM audit",
            "purge_age": "DELETE FROM audit WHERE ts < ?",
            "purge_size": "DELETE FROM audit WHERE id IN"
                          " (SELECT id FROM audit ORDER BY ts ASC, id ASC LIMIT ?)",
        },
    }
    _NO_LOWER = 0.0
    _NO_UPPER = 4e12          # year ~2096; comfortably above any real timestamp

    def _query(self, table: str, eq: tuple[str | None, str | None], *,
               limit: int, offset: int, since: float | None,
               until: float | None) -> dict:
        if self._conn is None:
            ring = self._recent_alerts if table == "alerts" else self._recent_audit
            items = _tail(ring, limit, offset)
            return {"total": len(items), "items": items, "degraded": True}

        spec = self._SQL[table]
        keys: tuple[str, ...] = spec["keys"]
        filters = [self._NO_LOWER if since is None else since,
                   self._NO_UPPER if until is None else until,
                   eq[0] or None, eq[0] or None,
                   eq[1] or None, eq[1] or None]
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(spec["count"], filters)
                total = int(cur.fetchone()[0])
                cur.execute(spec["page"],
                            [*filters, max(1, min(int(limit), 1000)), max(0, int(offset))])
                items = [{"timestamp": _iso(r["ts"]), **{k: r[k] for k in keys}}
                         for r in cur.fetchall()]
            return {"total": total, "items": items, "degraded": False}
        except sqlite3.Error as exc:
            logger.error("event store read failed on %s (%s)", table, exc)
            self.write_errors += 1
            return {"total": 0, "items": [], "degraded": True}

    def count_alerts(self, *, since: float | None = None, until: float | None = None) -> int:
        """Alert count in a time window, used by the metrics and stats endpoints."""
        if self._conn is None:
            return len(self._recent_alerts)
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(self._SQL["alerts"]["count"],
                            [self._NO_LOWER if since is None else since,
                             self._NO_UPPER if until is None else until,
                             None, None, None, None])
                return int(cur.fetchone()[0])
        except sqlite3.Error as exc:
            logger.error("event store count failed (%s)", exc)
            return 0

    def flush(self, timeout: float = 2.0) -> bool:
        """Block until the queue has drained (used by tests and clean shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cond:
                if not self._queue:
                    return True
            time.sleep(0.02)
        return False

    def stats(self) -> dict:
        return {
            "path": str(self.path),
            "degraded": self.degraded,
            "open_failed": self._open_failed,
            "written": self.written,
            "dropped": self.dropped,
            "write_errors": self.write_errors,
            "purged": self.purged,
            "pending": len(self._queue),
        }

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Stop the writer, flush what is still queued, close the connection."""
        if self._closed:
            return
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("event store writer did not stop within 5s")
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
