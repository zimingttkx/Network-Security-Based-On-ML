"""Packet parser: converts raw bytes / scapy packets to PacketInfo."""

from __future__ import annotations

import socket
import struct

from networksecurity.engine.detector import PacketInfo


class PacketParser:
    """Parse raw IPv4 TCP/UDP packets into PacketInfo.

    Supports raw bytes (from nfqueue / pcap) and scapy-style dicts.
    """

    @staticmethod
    def from_raw(data: bytes, timestamp: float = 0.0) -> PacketInfo | None:
        """Parse a raw IPv4 packet. Returns None on parse failure."""
        if len(data) < 20:
            return None

        version_ihl = data[0]
        if (version_ihl >> 4) != 4:
            return None  # IPv4 only

        ihl = (version_ihl & 0x0F) * 4
        total_len = struct.unpack("!H", data[2:4])[0]
        protocol = data[9]
        ttl = data[8]
        src_ip = socket.inet_ntoa(data[12:16])
        dst_ip = socket.inet_ntoa(data[16:20])

        src_port = dst_port = tcp_flags = 0

        transport_header_len = 8  # default UDP
        if protocol == 6:  # TCP
            # Read TCP data offset (top 4 bits of byte 12 of TCP header)
            tcp_data_offset = ((data[ihl + 12] >> 4) * 4) if len(data) > ihl + 12 else 20
            transport_header_len = tcp_data_offset
            if len(data) >= ihl + 20:
                src_port = struct.unpack("!H", data[ihl:ihl + 2])[0]
                dst_port = struct.unpack("!H", data[ihl + 2:ihl + 4])[0]
                tcp_flags = data[ihl + 13] & 0x3F
        elif protocol == 17 and len(data) >= ihl + 8:  # UDP
            src_port = struct.unpack("!H", data[ihl:ihl + 2])[0]
            dst_port = struct.unpack("!H", data[ihl + 2:ihl + 4])[0]

        return PacketInfo(
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            protocol=protocol,
            packet_size=total_len,
            timestamp=timestamp or 0.0,
            tcp_flags=tcp_flags,
            ttl=ttl,
            payload_size=max(0, total_len - ihl - transport_header_len),
        )

    @staticmethod
    def from_dict(d: dict) -> PacketInfo:
        return PacketInfo(
            src_ip=d.get("src_ip", "0.0.0.0"),
            dst_ip=d.get("dst_ip", "0.0.0.0"),
            src_port=d.get("src_port", 0),
            dst_port=d.get("dst_port", 0),
            protocol=d.get("protocol", 6),
            packet_size=d.get("packet_size", 0),
            timestamp=d.get("timestamp", 0.0),
            src_mac=d.get("src_mac", ""),
            dst_mac=d.get("dst_mac", ""),
            tcp_flags=d.get("tcp_flags", 0),
            ttl=d.get("ttl", 64),
            payload_size=d.get("payload_size", 0),
        )
