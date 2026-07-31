"""Network flow feature extraction from raw packets."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

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
    start_time: float = 0.0
    packet_count: int = 0
    byte_count: int = 0
    pkt_rate: float = 0.0
    mean_pkt_size: float = 0.0
    tcp_flags_or: int = 0
    tcp_flags_count: dict = field(default_factory=dict)

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
    """Tracks live 5-tuple flows and emits FlowFeatures on expiry."""

    def __init__(self, idle_timeout: float = 60.0, max_duration: float = 300.0):
        self._idle_timeout = idle_timeout
        self._max_duration = max_duration
        self._flows: dict[tuple, FlowFeatures] = {}
        self._last_seen: dict[tuple, float] = {}

    def ingest(self, packet: PacketInfo) -> Optional[FlowFeatures]:
        """Feed a packet. Returns a completed FlowFeatures or None."""
        key = (packet.src_ip, packet.dst_ip,
               packet.src_port, packet.dst_port, packet.protocol)
        now = packet.timestamp

        # Expire stale flows
        expired = None
        for k in list(self._flows):
            if now - self._last_seen.get(k, now) > self._idle_timeout:
                expired = self._flows.pop(k)
                self._last_seen.pop(k, None)
                break

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
        if flow.start_time == 0.0:
            flow.start_time = now
        flow.duration = now - flow.start_time
        flow.pkt_rate = (flow.packet_count / max(0.001, flow.duration)
                         if flow.duration else 0.0)
        flow.mean_pkt_size = flow.byte_count / max(1, flow.packet_count)
        flow.tcp_flags_or |= packet.tcp_flags

        if flow.duration > self._max_duration:
            return self._flows.pop(key)
        return expired

    def flush(self) -> list[FlowFeatures]:
        result = list(self._flows.values())
        self._flows.clear()
        self._last_seen.clear()
        return result
