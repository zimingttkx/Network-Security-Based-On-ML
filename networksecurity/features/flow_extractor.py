"""Network flow feature extraction from raw packets."""

from __future__ import annotations

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

    Wires into the detection pipeline via ``track()`` which ingests a
    ``PacketInfo`` and returns a completed ``FlowFeatures`` (when a flow hits
    its max duration) or ``None``.  The pipeline can use the returned vector
    as an additional statistical signal alongside AfterImage/LUCID.
    """

    def __init__(self, idle_timeout: float = 60.0, max_duration: float = 300.0,
                 max_flows: int = 100000):
        self._idle_timeout = idle_timeout
        self._max_duration = max_duration
        # Hard cap on concurrently tracked flows.  Under a spoofed-source flood
        # every packet creates a new 5-tuple; without a cap ``_flows`` grows
        # without bound (memory) and the idle sweep scans the whole table every
        # packet (O(N) per packet -> O(N^2) aggregate).  At the cap, the
        # least-recently-seen flow is evicted (buffered for emission) instead.
        self._max_flows = max(1, max_flows)
        self._flows: dict[tuple, FlowFeatures] = {}
        self._last_seen: dict[tuple, float] = {}
        # Flows evicted by idle timeout that could not be returned on the
        # triggering packet (because only one FlowFeatures may be returned).
        # They are buffered and emitted on subsequent track() calls.
        self._pending: list[FlowFeatures] = []

    def track(self, packet: PacketInfo) -> FlowFeatures | None:
        """Feed a packet. Returns a completed FlowFeatures or None."""
        completed = self.ingest(packet)
        return completed

    def ingest(self, packet: PacketInfo) -> FlowFeatures | None:
        """Feed a packet. Returns a completed FlowFeatures or None.

        Priority of what is returned on a given call:
          1. The current flow if it completes via its own max_duration.
          2. A previously buffered idle-expired flow (``_pending``).

        Only one FlowFeatures is returned per call to honour the contract, but
        every completed/evicted flow is guaranteed to be emitted eventually via
        the buffer.
        """
        key = (packet.src_ip, packet.dst_ip,
               packet.src_port, packet.dst_port, packet.protocol)
        now = packet.timestamp

        # Sweep stale flows.  The table is capped (max_flows), so a full sweep
        # is bounded work; the current key, if stale, must NOT be popped here:
        # it is about to be refreshed below, and popping it would corrupt its
        # aggregation. All other expired flows are buffered for later emission
        # (a call may only return one FlowFeatures, so they queue up).
        swept: list[FlowFeatures] = []
        for k in list(self._flows):
            if k == key:
                continue
            if now - self._last_seen.get(k, now) > self._idle_timeout:
                evicted = self._flows.pop(k)
                self._last_seen.pop(k, None)
                swept.append(evicted)

        # Enforce the flow-table cap: evict the least-recently-seen flow when
        # a flood of new 5-tuples pushes the table over budget.
        while len(self._flows) >= self._max_flows and key not in self._flows:
            lru_key = min(self._flows, key=lambda k: self._last_seen.get(k, 0.0))
            evicted = self._flows.pop(lru_key)
            self._last_seen.pop(lru_key, None)
            swept.append(evicted)

        if key not in self._flows:
            self._flows[key] = FlowFeatures(
                src_ip=packet.src_ip, dst_ip=packet.dst_ip,
                src_port=packet.src_port, dst_port=packet.dst_port,
                protocol=packet.protocol,
            )

        flow = self._flows[key]
        self._last_seen[key] = now
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

        # Buffer any idle-expired flows swept by THIS call BEFORE checking the
        # current flow's max_duration. This guarantees they are never silently
        # dropped when the current flow completes via max_duration (which returns
        # early below, skipping the extend that would otherwise follow it).
        self._pending.extend(swept)

        # 1) The current flow completes via max_duration — emit it now.
        #    Checked AFTER _pending is topped up, so a flow reaching its max
        #    duration is returned promptly (not starved by the backlog) AND the
        #    idle-expired flows swept here are already buffered for later calls.
        if flow.duration > self._max_duration:
            self._flows.pop(key)
            self._last_seen.pop(key, None)  # keep both maps in sync — leak fix
            return flow

        # 2) Emit a previously buffered idle-expired flow.
        if self._pending:
            return self._pending.pop(0)

        return None

    def flush(self) -> list[FlowFeatures]:
        result = list(self._flows.values()) + list(self._pending)
        self._flows.clear()
        self._last_seen.clear()
        self._pending.clear()
        return result

    @property
    def active_flow_count(self) -> int:
        return len(self._flows)
