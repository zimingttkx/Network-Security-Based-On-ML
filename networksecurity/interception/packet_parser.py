"""Packet parser: converts raw bytes / scapy packets to PacketInfo."""

from __future__ import annotations

import socket
import struct

from networksecurity.engine.detector import PacketInfo

# IPv4 / L4 header bounds in bytes.
_MIN_IP_HEADER = 20
_MAX_IP_HEADER = 60
_TCP_MIN_HEADER = 20
_TCP_MAX_HEADER = 60
_UDP_HEADER = 8

# The live NFQUEUE hook sits on INPUT (see IptablesManager.setup_nfqueue), so
# every packet reaching from_raw is inbound.
_LIVE_DIRECTION = 1


class PacketParser:
    """Parse raw IPv4 TCP/UDP packets into PacketInfo.

    Supports raw bytes (from nfqueue / pcap) and scapy-style dicts.

    ``from_raw`` is fail-closed: anything it cannot parse completely and
    consistently returns None instead of a PacketInfo assembled from
    half-validated offsets.  The caller drops those packets and counts them.
    """

    @staticmethod
    def from_raw(data: bytes, timestamp: float = 0.0) -> PacketInfo | None:
        """Parse a raw IPv4 packet.  Returns None on any parse failure.

        A PacketInfo built from a bogus offset is worse than no PacketInfo at
        all: the ports would be arbitrary bytes of the IP header, of a
        fragment's payload, or past the end of a truncated capture, and the
        whole detection chain would then rate-limit, blacklist and build
        autoencoder features over values that mean nothing.  Every length and
        offset below is therefore cross-checked against both ``len(data)``
        (bytes actually captured) and ``total_len`` (what the header claims),
        and any inconsistency returns None.

        Note that ``total_len > len(data)`` alone is *not* rejected — a capture
        truncated by snaplen still carries valid headers, and ``total_len`` is
        the authoritative on-wire size for ``packet_size``/``payload_size``.
        """
        if len(data) < _MIN_IP_HEADER:
            return None

        version_ihl = data[0]
        if (version_ihl >> 4) != 4:
            return None  # IPv4 only

        ihl = (version_ihl & 0x0F) * 4
        # An IHL below 5 means the fixed header itself is incomplete, and every
        # L4 offset computed from it would land inside the IP header.
        if ihl < _MIN_IP_HEADER or ihl > _MAX_IP_HEADER or ihl > len(data):
            return None

        total_len = struct.unpack("!H", data[2:4])[0]
        if total_len < ihl:
            return None  # packet shorter than its own IP header

        # Non-first fragments carry no L4 header at all — the bytes at `ihl`
        # are a continuation of an earlier fragment's payload, so reading ports
        # there yields arbitrary values.  Dropping them is the fail-closed
        # choice; the cost (legitimate large fragmented UDP is not inspected)
        # is documented in SECURITY.md and observable via parse_failed_count.
        if struct.unpack("!H", data[6:8])[0] & 0x1FFF:
            return None

        protocol = data[9]
        ttl = data[8]
        src_ip = socket.inet_ntoa(data[12:16])
        dst_ip = socket.inet_ntoa(data[16:20])

        src_port = dst_port = tcp_flags = window_size = 0
        transport_header_len = 0

        if protocol == 6:  # TCP
            if (len(data) < ihl + _TCP_MIN_HEADER
                    or total_len < ihl + _TCP_MIN_HEADER):
                return None
            data_offset = (data[ihl + 12] >> 4) * 4
            if data_offset < _TCP_MIN_HEADER or data_offset > _TCP_MAX_HEADER:
                return None
            # The header claims options we must actually have before reading
            # anything past byte 20 of the TCP header.
            if len(data) < ihl + data_offset or total_len < ihl + data_offset:
                return None
            src_port = struct.unpack("!H", data[ihl:ihl + 2])[0]
            dst_port = struct.unpack("!H", data[ihl + 2:ihl + 4])[0]
            tcp_flags = data[ihl + 13] & 0x3F
            window_size = struct.unpack("!H", data[ihl + 14:ihl + 16])[0]
            transport_header_len = data_offset
        elif protocol == 17:  # UDP
            if len(data) < ihl + _UDP_HEADER or total_len < ihl + _UDP_HEADER:
                return None
            src_port = struct.unpack("!H", data[ihl:ihl + 2])[0]
            dst_port = struct.unpack("!H", data[ihl + 2:ihl + 4])[0]
            transport_header_len = _UDP_HEADER
        # Any other protocol keeps ports/window at 0 and is still returned: the
        # rule engine's allowed_protocols filter blocks it, which is a more
        # auditable outcome than a silent parse failure.

        return PacketInfo(
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            protocol=protocol,
            packet_size=total_len,
            timestamp=timestamp or 0.0,
            tcp_flags=tcp_flags,
            ttl=ttl,
            payload_size=max(0, total_len - ihl - transport_header_len),
            window_size=window_size,
            direction=_LIVE_DIRECTION,
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
            window_size=d.get("window_size", 0),
            direction=d.get("direction", 0),
        )
