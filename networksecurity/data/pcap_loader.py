"""Pcap file loader — reads pcap/pcapng files for offline testing."""

from __future__ import annotations

import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class PcapLoader:
    """Load packets from pcap files.

    Requires scapy: pip install scapy
    """

    def __init__(self) -> None:
        self._scapy = None

    async def load(self, path: str) -> AsyncIterator[dict]:
        """Yield packet dicts from a pcap file."""
        try:
            from scapy.utils import RawPcapReader  # type: ignore
            from scapy.layers.inet import IP, TCP, UDP  # type: ignore
            from scapy.layers.l2 import Ether  # type: ignore
        except ImportError:
            raise ImportError(
                "scapy is required for pcap loading. Install: pip install scapy"
            )

        count = 0
        for pkt_data, metadata in RawPcapReader(path):
            count += 1
            try:
                from scapy.compat import raw
                pkt = Ether(raw(pkt_data))
            except Exception:
                yield self._empty_packet(count)
                continue

            ip_layer = pkt.getlayer(IP)
            if not ip_layer:
                yield self._empty_packet(count)
                continue

            tcp_layer = pkt.getlayer(TCP)
            udp_layer = pkt.getlayer(UDP)

            yield {
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "src_port": tcp_layer.sport if tcp_layer else (udp_layer.sport if udp_layer else 0),
                "dst_port": tcp_layer.dport if tcp_layer else (udp_layer.dport if udp_layer else 0),
                "protocol": 6 if tcp_layer else (17 if udp_layer else ip_layer.proto),
                "packet_size": len(pkt_data),
                "timestamp": float(metadata.sec) + float(metadata.usec) / 1_000_000,
                "src_mac": pkt.src if hasattr(pkt, "src") else "",
                "dst_mac": pkt.dst if hasattr(pkt, "dst") else "",
                "tcp_flags": tcp_layer.flags if tcp_layer else 0,
                "ttl": ip_layer.ttl,
                "payload_size": len(pkt_data) - (ip_layer.ihl * 4) - (20 if tcp_layer else 8 if udp_layer else 0),
            }

        logger.info("Loaded %d packets from %s", count, path)

    @staticmethod
    def _empty_packet(n: int) -> dict:
        return {
            "src_ip": "0.0.0.0", "dst_ip": "0.0.0.0",
            "src_port": 0, "dst_port": 0, "protocol": 0,
            "packet_size": 0, "timestamp": 0.0,
            "src_mac": "", "dst_mac": "", "tcp_flags": 0,
            "ttl": 0, "payload_size": 0,
        }
