"""Pcap file loader — reads pcap/pcapng files for offline testing."""

from __future__ import annotations

import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class PcapLoader:
    """Load packets from pcap files.

    Requires scapy: pip install scapy

    Interface contract
    ------------------
    Yields ``dict`` for every packet that parses as IPv4 (with the same
    field names as ``PacketParser.from_dict``), or ``None`` for frames that
    cannot be parsed (non-IPv4 such as ARP, malformed frames).  ``None``
    entries mirror the ``None`` returned by ``PacketParser.from_raw`` for
    unparseable live packets, so callers can skip them uniformly.

    ``packet_size`` / ``payload_size`` follow the **IP-layer** convention
    (``ip_total`` and ``ip_total - ip_hdr - l4_hdr``) to stay consistent with
    the live ``PacketParser.from_raw`` path, which reports IP total length.
    """

    def __init__(self) -> None:
        self._scapy = None

    async def load(self, path: str) -> AsyncIterator[dict | None]:
        """Yield packet dicts (or None for unparseable frames) from a pcap file."""
        try:
            from scapy.utils import RawPcapReader  # type: ignore
            from scapy.layers.inet import IP, TCP, UDP  # type: ignore
            from scapy.layers.l2 import Ether  # type: ignore
        except ImportError:
            raise ImportError(
                "scapy is required for pcap loading. Install: pip install scapy"
            )

        count = 0
        parsed = 0
        for pkt_data, metadata in RawPcapReader(path):
            count += 1
            try:
                from scapy.compat import raw
                pkt = Ether(raw(pkt_data))
            except Exception:
                yield None
                continue

            ip_layer = pkt.getlayer(IP)
            if not ip_layer:
                # Non-IPv4 frame (e.g. ARP) — unparseable for our pipeline.
                yield None
                continue

            tcp_layer = pkt.getlayer(TCP)
            udp_layer = pkt.getlayer(UDP)

            # Compute payload size from the IP length (link-layer headers like
            # Ethernet/802.1Q are excluded, so the result is the true L4 payload).
            ip_total = int(ip_layer.len)
            l4_header = (
                int(tcp_layer.dataofs) * 4 if tcp_layer
                else 8 if udp_layer
                else 0
            )
            header_len = int(ip_layer.ihl) * 4
            payload_size = max(0, ip_total - header_len - l4_header)

            yield {
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "src_port": int(tcp_layer.sport) if tcp_layer else (int(udp_layer.sport) if udp_layer else 0),
                "dst_port": int(tcp_layer.dport) if tcp_layer else (int(udp_layer.dport) if udp_layer else 0),
                # protocol is taken from the IP layer so it stays correct even
                # when only one of TCP/UDP is present.
                "protocol": int(ip_layer.proto),
                # IP-layer packet size (matches PacketParser.from_raw's total_len),
                # kept distinct from `payload_size` which is the L4 payload.
                "packet_size": ip_total,
                "timestamp": float(metadata.sec) + float(metadata.usec) / 1_000_000,
                "src_mac": pkt.src if hasattr(pkt, "src") else "",
                "dst_mac": pkt.dst if hasattr(pkt, "dst") else "",
                # scapy returns a FlagValue (NOT a plain int); coerce to int so
                # downstream consumers expecting an int (numpy arrays, FlowTracker
                # bitwise-or, JSON serialization) behave consistently.
                "tcp_flags": int(tcp_layer.flags) if tcp_layer else 0,
                "ttl": int(ip_layer.ttl),
                "payload_size": payload_size,
            }
            parsed += 1

        logger.info("Loaded %d packets from %s (%d parsed as IPv4)", parsed, path, count)

    @staticmethod
    def _empty_packet(n: int) -> dict:
        return {
            "src_ip": "0.0.0.0", "dst_ip": "0.0.0.0",
            "src_port": 0, "dst_port": 0, "protocol": 0,
            "packet_size": 0, "timestamp": 0.0,
            "src_mac": "", "dst_mac": "", "tcp_flags": 0,
            "ttl": 0, "payload_size": 0,
        }

    @staticmethod
    def _empty_packet(n: int) -> dict:
        return {
            "src_ip": "0.0.0.0", "dst_ip": "0.0.0.0",
            "src_port": 0, "dst_port": 0, "protocol": 0,
            "packet_size": 0, "timestamp": 0.0,
            "src_mac": "", "dst_mac": "", "tcp_flags": 0,
            "ttl": 0, "payload_size": 0,
        }
