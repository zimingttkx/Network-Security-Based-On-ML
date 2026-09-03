"""Network flow feature extraction from raw packets."""

from __future__ import annotations

import heapq
import itertools
from collections import OrderedDict
from dataclasses import dataclass

from networksecurity.engine.detector import PacketInfo


@dataclass
class FlowFeatures:
    """Statistical features computed over a 5-tuple flow."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    duration: float = 0.0
    start_time: float | None = None
    packet_count: int = 0
    byte_count: int = 0
    pkt_rate: float = 0.0
    mean_pkt_size: float = 0.0
    tcp_flags_or: int = 0

    def to_vector(self) -> list[float]:
        return [
            self.duration,
            float(self.packet_count),
            float(self.byte_count),
            self.pkt_rate,
            self.mean_pkt_size,
            float(self.protocol),
            float(self.src_port) / 65535.0,
            float(self.dst_port) / 65535.0,
            float(self.tcp_flags_or) / 255.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "duration", "packet_count", "byte_count", "pkt_rate",
            "mean_pkt_size", "protocol", "src_port_norm", "dst_port_norm",
            "tcp_flags_or",
        ]


class FlowTracker:
    """Tracks live 5-tuple flows and emits FlowFeatures on expiry.

    Data structures (chosen for bounded per-packet work under flood):
    - ``_flows`` is an OrderedDict used as an LRU: ``move_to_end`` on touch,
      ``popitem(last=False)`` evicts the least-recently-seen flow in O(1)
      when the table cap is hit.
    - ``_expiry_heap`` is a min-heap of ``(idle_deadline, tiebreak, key)``
      so the idle sweep only pops entries that are ACTUALLY expired
      (O(k log n), k = expired count) instead of scanning the whole table
      every packet.  Stale heap entries (key re-touched or already evicted)
      are skipped lazily via a generation counter.

    Wires into the detection pipeline via ``track()`` which ingests a
    ``PacketInfo`` and returns a completed ``FlowFeatures`` (when a flow hits
    its max duration) or ``None``.
    """

    def __init__(self, idle_timeout: float = 60.0, max_duration: float = 300.0,
                 max_flows: int = 100000):
        self._idle_timeout = idle_timeout
        self._max_duration = max_duration
        # Hard cap on concurrently tracked flows.  Under a spoofed-source flood
        # every packet creates a new 5-tuple; without a cap ``_flows`` grows
        # without bound (memory).  At the cap, the least-recently-seen flow is
        # evicted (buffered for emission) instead.
        self._max_flows = max(1, max_flows)
        self._flows: "OrderedDict[tuple, FlowFeatures]" = OrderedDict()
        self._last_seen: dict[tuple, float] = {}
        # Min-heap of (deadline, gen, key); ``gen`` disambiguates re-inserted
        # keys so stale entries are identified by comparing against the
        # key's current generation (see _sweep_expired).
        self._expiry_heap: list[tuple[float, int, tuple]] = []
        self._gen: dict[tuple, int] = {}
        self._heap_tie = itertools.count()
        # Flows evicted by idle timeout that could not be returned on the
        # triggering packet (because only one FlowFeatures may be returned).
        # They are buffered and emitted on subsequent track() calls.  The
        # buffer itself is capped: under a flood the sweeper can produce more
        # evictions than the caller consumes, and an uncapped list here was
        # measured at >500 MB RSS.
        self._pending: list[FlowFeatures] = []
        self._pending_max = max(1, self._max_flows // 2)
        self._pending_dropped = 0

    def track(self, packet: PacketInfo) -> FlowFeatures | None:
        """Feed a packet. Returns a completed FlowFeatures or None."""
        completed = self.ingest(packet)
        return completed

    def ingest(self, packet: PacketInfo) -> FlowFeatures | None:
        """Feed a packet. Returns a completed FlowFeatures or None.

        Priority of what is returned on a given call:
          1. The current flow if it completes via its own max_duration.
          2. A previously buffered idle-expired flow (``_pending``).

        Only one FlowFeatures is returned per call to honour the contract.
        Swept/evicted flows beyond that are buffered (capped); when the
        buffer overflows the OLDEST buffered flow is dropped and counted in
        ``pending_dropped`` (visible via status()).
        """
        key = (packet.src_ip, packet.dst_ip,
               packet.src_port, packet.dst_port, packet.protocol)
        now = packet.timestamp

        # Idle sweep: pop only heap entries whose deadline has actually
        # passed (bounded by the number expired, not the table size).  The
        # CURRENT key is never swept here — it is about to be refreshed.
        self._sweep_expired(now, keep=key)

        # Enforce the table cap: O(1) LRU eviction via popitem(last=False).
        swept: list[FlowFeatures] = []
        while len(self._flows) >= self._max_flows and key not in self._flows:
            lru_key, evicted = self._flows.popitem(last=False)
            self._last_seen.pop(lru_key, None)
            self._gen.pop(lru_key, None)
            swept.append(evicted)

        if key not in self._flows:
            self._flows[key] = FlowFeatures(
                src_ip=packet.src_ip, dst_ip=packet.dst_ip,
                src_port=packet.src_port, dst_port=packet.dst_port,
                protocol=packet.protocol,
            )
        else:
            self._flows.move_to_end(key)  # LRU refresh

        flow = self._flows[key]
        self._last_seen[key] = now
        # (Re)schedule this flow's idle deadline: bump its generation so any
        # older heap entry for the same key becomes a no-op when reached.
        gen = next(self._heap_tie)
        self._gen[key] = gen
        heapq.heappush(self._expiry_heap,
                       (now + self._idle_timeout, gen, key))

        flow.packet_count += 1
        flow.byte_count += packet.packet_size
        if flow.start_time is None:
            flow.start_time = now
        # Guard against out-of-order / non-monotonic timestamps: never let
        # duration go negative (would yield nonsensical negative duration and
        # absurd pkt_rate). The flow's clock is monotonic relative to its start.
        flow.duration = max(0.0, now - flow.start_time)
        flow.pkt_rate = flow.packet_count / max(0.001, flow.duration)
        flow.mean_pkt_size = flow.byte_count / max(1, flow.packet_count)
        flow.tcp_flags_or |= packet.tcp_flags

        # Buffer swept/evicted flows BEFORE checking the current flow's
        # max_duration so they are never silently dropped by the early
        # return below.
        if swept:
            self._buffer(swept)

        # 1) The current flow completes via max_duration — emit it now.
        if flow.duration > self._max_duration:
            self._flows.pop(key, None)
            self._last_seen.pop(key, None)
            self._gen.pop(key, None)  # stale heap entry skipped on reach
            return flow

        # 2) Emit a previously buffered idle-expired flow.
        if self._pending:
            return self._pending.pop(0)

        return None

    # -- internals -----------------------------------------------------------

    def _sweep_expired(self, now: float, keep: tuple | None = None) -> None:
        """Pop idle-expired flows off the expiry heap.

        Only heap entries whose deadline <= now are examined, so the cost per
        packet is O(expired log n), NOT O(table).  A heap entry is stale if
        its generation differs from the key's current one (the flow was
        re-touched after the entry was pushed) or the key is gone (flow
        completed/evicted earlier) — those are skipped, not acted on.
        """
        expired: list[FlowFeatures] = []
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            deadline, gen, key = heapq.heappop(self._expiry_heap)
            if key == keep or self._gen.get(key) != gen:
                continue  # refreshed or completed/evicted — entry is stale
            flow = self._flows.pop(key, None)
            if flow is None:
                continue
            self._last_seen.pop(key, None)
            self._gen.pop(key, None)
            expired.append(flow)
        if expired:
            self._buffer(expired)

    def _buffer(self, flows: list[FlowFeatures]) -> None:
        """Append to the pending buffer with a hard cap (drop oldest)."""
        if len(self._pending) + len(flows) > self._pending_max:
            keep_n = max(0, self._pending_max - len(self._pending))
            self._pending_dropped += len(flows) - keep_n
            flows = flows[:keep_n]
        self._pending.extend(flows)

    def flush(self) -> list[FlowFeatures]:
        result = list(self._flows.values()) + list(self._pending)
        self._flows.clear()
        self._last_seen.clear()
        self._expiry_heap.clear()
        self._gen.clear()
        self._pending.clear()
        return result

    def status(self) -> dict:
        """Tracker internals for dashboards/tests."""
        return {
            "active_flows": len(self._flows),
            "pending_buffered": len(self._pending),
            "pending_dropped": self._pending_dropped,
            "max_flows": self._max_flows,
        }

    @property
    def active_flow_count(self) -> int:
        return len(self._flows)
