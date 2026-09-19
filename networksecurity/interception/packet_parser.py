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
# ICMP type + code.  The remaining 4 bytes of the ICMP header (id/seq or unused)
# carry no rule-relevant field here, so only this much is required to be present.
_ICMP_HEADER = 4

# IPv6: fixed header is exactly 40 bytes and has no IHL, so the upper-layer
# offset depends entirely on walking the extension-header chain.
_IPV6_HEADER = 40
# Chain depth cap.  Real traffic carries one or two; anything deeper is either a
# loop or a crafted header set, and walking it to "see where it ends" is exactly
# what parser-evasion payloads bet on.
_MAX_IPV6_HEADERS = 6
# Headers we stop at and hand to the transport parser.  59 (No Next Header) is
# included so the walk ends and the rule engine reports the packet as an
# unallowed protocol instead of the parser guessing.
_IPV6_TRANSPORT_HEADERS = frozenset({6, 17, 58, 59})
# Extension headers whose length is fixed and known without reading them.
_IPV6_FIXED_LENGTHS = {44: 8}          # Fragment header: always 8 bytes
# AH/ESP carry no parseable next-header chain without authenticating the packet,
# and their length fields describe ciphertext.  Refused rather than skipped:
# continuing the walk past them would read encrypted bytes as a TCP header.
_IPV6_UNPARSEABLE = frozenset({50, 51})

# The live NFQUEUE hook sits on INPUT (see IptablesManager.setup_nfqueue), so
# every packet reaching from_raw is inbound.
_LIVE_DIRECTION = 1


class PacketParser:
    """Parse raw IPv4/IPv6 TCP/UDP/ICMP packets into PacketInfo.

    Supports raw bytes (from nfqueue / pcap) and scapy-style dicts.

    ``from_raw`` is fail-closed: anything it cannot parse completely and
    consistently returns None instead of a PacketInfo assembled from
    half-validated offsets.  The caller drops those packets and counts them.
    """

    @staticmethod
    def _parse_transport(protocol: int, data: bytes, offset: int,
                         total_len: int) -> tuple | None:
        """Read the upper-layer header at *offset*.

        Returns ``(src_port, dst_port, tcp_flags, window, header_len,
        icmp_type, icmp_code)`` or None when the header is missing, truncated
        or internally inconsistent.  Shared by both address families so IPv6
        cannot drift into weaker validation than IPv4.
        """
        src_port = dst_port = tcp_flags = window_size = 0
        icmp_type = icmp_code = 0
        transport_header_len = 0

        if protocol == 6:  # TCP
            if (len(data) < offset + _TCP_MIN_HEADER
                    or total_len < offset + _TCP_MIN_HEADER):
                return None
            data_offset = (data[offset + 12] >> 4) * 4
            if data_offset < _TCP_MIN_HEADER or data_offset > _TCP_MAX_HEADER:
                return None
            # The header claims options we must actually have before reading
            # anything past byte 20 of the TCP header.
            if len(data) < offset + data_offset or total_len < offset + data_offset:
                return None
            src_port = struct.unpack("!H", data[offset:offset + 2])[0]
            dst_port = struct.unpack("!H", data[offset + 2:offset + 4])[0]
            tcp_flags = data[offset + 13] & 0x3F
            window_size = struct.unpack("!H", data[offset + 14:offset + 16])[0]
            transport_header_len = data_offset
        elif protocol == 17:  # UDP
            if len(data) < offset + _UDP_HEADER or total_len < offset + _UDP_HEADER:
                return None
            src_port = struct.unpack("!H", data[offset:offset + 2])[0]
            dst_port = struct.unpack("!H", data[offset + 2:offset + 4])[0]
            transport_header_len = _UDP_HEADER
        elif protocol in (1, 58):  # ICMP / ICMPv6: type and code drive the policy
            if (len(data) < offset + _ICMP_HEADER
                    or total_len < offset + _ICMP_HEADER):
                return None
            icmp_type = data[offset]
            icmp_code = data[offset + 1]
            transport_header_len = _ICMP_HEADER
        # Any other protocol keeps ports/window at 0 and is still returned: the
        # rule engine's allowed_protocols filter blocks it, which is a more
        # auditable outcome than a silent parse failure.

        return (src_port, dst_port, tcp_flags, window_size,
                transport_header_len, icmp_type, icmp_code)

    @staticmethod
    def _parse_ipv6(data: bytes, timestamp: float) -> PacketInfo | None:
        """Fixed 40-byte header, then walk the extension-header chain.

        Every hop is length-checked against the captured bytes *and* the
        payload length the header declares.  A chain that loops, exceeds
        ``_MAX_IPV6_HEADERS`` or runs past the end of the packet returns None
        rather than guessing an offset: extension headers are the classic
        parser-evasion surface, and a bogus offset poisons every downstream
        feature and rule match exactly like a malformed IPv4 header does.
        """
        if len(data) < _IPV6_HEADER:
            return None
        if (data[0] >> 4) != 6:
            return None

        payload_len = struct.unpack("!H", data[4:6])[0]
        total_len = _IPV6_HEADER + payload_len
        if total_len < _IPV6_HEADER:
            return None

        next_header = data[6]
        hop_limit = data[7]
        src_ip = socket.inet_ntop(socket.AF_INET6, data[8:24])
        dst_ip = socket.inet_ntop(socket.AF_INET6, data[24:40])

        offset = _IPV6_HEADER
        for _ in range(_MAX_IPV6_HEADERS):
            if next_header in _IPV6_TRANSPORT_HEADERS:
                break
            header_len = _IPV6_FIXED_LENGTHS.get(next_header)
            if header_len is None:
                if next_header in _IPV6_UNPARSEABLE:
                    return None  # ESP/AH payloads: the chain ends in ciphertext
                if len(data) < offset + 2 or total_len < offset + 2:
                    return None
                header_len = (data[offset + 1] + 1) * 8
            if len(data) < offset + header_len or total_len < offset + header_len:
                return None
            if next_header == 44:  # Fragment
                if header_len < 8:
                    return None
                # frag offset occupies bits 3..15 of the third/fourth bytes; a
                # non-first fragment carries no upper header at all.
                if struct.unpack("!H", data[offset + 2:offset + 4])[0] & 0xFFF8:
                    return None
            next_header = data[offset]
            offset += header_len
        else:
            return None  # chain too deep: treat as crafted, not as unusual

        transport = PacketParser._parse_transport(next_header, data, offset, total_len)
        if transport is None:
            return None
        src_port, dst_port, tcp_flags, window_size, transport_len, icmp_type, icmp_code = transport

        return PacketInfo(
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            protocol=next_header,
            packet_size=total_len,
            timestamp=timestamp or 0.0,
            tcp_flags=tcp_flags,
            ttl=hop_limit,
            payload_size=max(0, total_len - offset - transport_len),
            window_size=window_size,
            direction=_LIVE_DIRECTION,
            icmp_type=icmp_type,
            icmp_code=icmp_code,
        )

    @staticmethod
    def from_raw(data: bytes, timestamp: float = 0.0) -> PacketInfo | None:
        """Parse a raw IPv4 or IPv6 packet.  Returns None on any parse failure.

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

        version = data[0] >> 4
        if version == 6:
            return PacketParser._parse_ipv6(data, timestamp)
        if version != 4:
            return None

        version_ihl = data[0]
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

        transport = PacketParser._parse_transport(protocol, data, ihl, total_len)
        if transport is None:
            return None
        src_port, dst_port, tcp_flags, window_size, transport_len, icmp_type, icmp_code = transport

        return PacketInfo(
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            protocol=protocol,
            packet_size=total_len,
            timestamp=timestamp or 0.0,
            tcp_flags=tcp_flags,
            ttl=ttl,
            payload_size=max(0, total_len - ihl - transport_len),
            window_size=window_size,
            direction=_LIVE_DIRECTION,
            icmp_type=icmp_type,
            icmp_code=icmp_code,
        )

    @staticmethod
    def from_dict(d: dict) -> PacketInfo:
        return PacketInfo(
            src_ip=d.get("src_ip", "0.0.0.0"),
            dst_ip=d.get("dst_ip", "0.0.0.0"),
            src_port=d.get("src_port", 0),
            dst_port=d.get("dst_port", 0),
            protocol=d.get("protocol", 6),
            icmp_type=d.get("icmp_type", 0),
            icmp_code=d.get("icmp_code", 0),
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
