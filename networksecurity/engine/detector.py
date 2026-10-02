"""Base detector interface and packet data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    # TCP advertised window; 0 for UDP and for non-TCP/UDP protocols.
    window_size: int = 0
    # Flow direction using LUCID's convention: 0 = outbound, 1 = inbound.
    direction: int = 0
    # ICMP type/code, 0 for every other protocol.  Rule fields only: deliberately
    # absent from to_dict(), so the AfterImage feature vector (and its 90-dim
    # contract) cannot shift because a header field was added here.
    icmp_type: int = 0
    icmp_code: int = 0

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
            "window_size": self.window_size,
            "direction": self.direction,
        }


class BaseDetector(ABC):
    """Contract every detection module implements.

    ``process_packet`` returns a Verdict, or ``None`` to abstain and let the
    next detector decide.  Any non-BLOCK verdict ends the chain; BLOCK ends it
    too unless the pipeline runs without short-circuiting.  A detector that is
    not ``ready`` must abstain — returning a verdict while unready would make
    the pipeline count it as having run, and that is what fail-closed decides
    on.

    Nothing may be opened, spawned or written at import time: construction and
    ``configure`` only, side effects start at the first packet.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._packet_count: int = 0

    @abstractmethod
    async def process_packet(self, packet: PacketInfo) -> Verdict | None:
        """Process a single packet.  Return a Verdict or None."""

    def configure(self, params: dict) -> None:
        """Apply tuning read from config.yaml, before the first packet.

        Unknown keys are rejected rather than ignored: a silently dropped
        option looks exactly like a detector that was configured.
        """
        if params:
            raise ValueError(f"{self.name} accepts no detector params, got {sorted(params)}")

    async def process_batch(self, packets: list[PacketInfo]) -> list[Verdict | None]:
        return [await self.process_packet(p) for p in packets]

    @property
    def ready(self) -> bool:
        """Whether this detector can actually score packets right now."""
        return True

    def status(self) -> dict:
        """Detector-owned fields for /api/v1/status; empty by default."""
        return {}

    @property
    def packet_count(self) -> int:
        return self._packet_count

    def reset(self) -> None:
        self._packet_count = 0
