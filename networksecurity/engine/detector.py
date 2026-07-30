"""Base detector interface and packet data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from networksecurity.engine.verdict import Verdict


@dataclass
class PacketInfo:
    """Normalized packet metadata from any capture source."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP
    packet_size: int
    timestamp: float
    src_mac: str = ""
    dst_mac: str = ""
    tcp_flags: int = 0
    ttl: int = 64
    payload_size: int = 0

    def to_dict(self) -> dict:
        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "packet_size": self.packet_size,
            "timestamp": self.timestamp,
            "src_mac": self.src_mac,
            "dst_mac": self.dst_mac,
            "tcp_flags": self.tcp_flags,
            "ttl": self.ttl,
            "payload_size": self.payload_size,
        }


class BaseDetector(ABC):
    """Abstract base for all detection modules.

    Each detector receives a packet and returns either a Verdict
    or None (meaning "pass to next detector").
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._packet_count: int = 0

    @abstractmethod
    async def process_packet(self, packet: PacketInfo) -> Optional[Verdict]:
        """Process a single packet.  Return a Verdict or None."""

    async def process_batch(self, packets: list[PacketInfo]) -> list[Optional[Verdict]]:
        return [await self.process_packet(p) for p in packets]

    @property
    def packet_count(self) -> int:
        return self._packet_count

    def reset(self) -> None:
        self._packet_count = 0
