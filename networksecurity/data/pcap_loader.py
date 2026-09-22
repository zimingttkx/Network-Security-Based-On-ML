"""Pcap file loader — reads pcap/pcapng files for offline testing."""

from __future__ import annotations

import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# scapy is imported lazily inside load()/_parse_ip() (only when they actually
# run) because it is a heavy optional dependency.


class PcapLoader:
    """Load packets from pcap files.

    Requires scapy: pip install scapy

    Interface contract
    ------------------
    Yields a ``dict`` for every packet that parses as IPv4 (with the same
    field names as ``PacketParser.from_dict``), or ``None`` for frames that
    are not IPv4-parseable (e.g. ARP, non-IPv4 link layers).  ``None``
    entries mirror the ``None`` returned by ``PacketParser.from_raw`` for
    unparseable live packets, so callers can skip them uniformly.

    For an IPv4 frame whose L4 segment is truncated or malformed (e.g. a
    snaplen-limited TCP dump, a non-first fragment, a ``dataofs`` outside
    20..60, or an IP total length that contradicts the captured byte count),
    the loader yields ``None`` — mirroring ``PacketParser.from_raw``, which is
    fail-closed on exactly the same conditions.  Emitting a best-effort record
    instead would make the two ingestion paths disagree about which frames are
    valid: a packet dropped live would still be trained on offline, so the
    model would be fitted to features the live path never produces.

    ``packet_size`` / ``payload_size`` follow the **IP-layer** convention
    (``ip_total`` and ``ip_total - ip_hdr - l4_hdr``) to stay consistent with
    the live ``PacketParser.from_raw`` path, which reports IP total length.

    Multiple link-layer types are handled: Ethernet (1), raw IP (101, e.g.
    VPN/tun interfaces), and Linux cooked capture SLL (113, e.g.
    ``tcpdump -i any``).  VLAN-tagged (802.1Q) Ethernet frames are parsed by
    scapy's ``Ether`` dissector as well.
    """

    async def load(self, path: str) -> AsyncIterator[dict | None]:
        """Yield packet dicts (or None for non-IPv4 frames) from a pcap file."""
        try:
            from scapy.layers.inet import TCP, UDP  # type: ignore
            from scapy.layers.l2 import Dot1Q, Ether  # type: ignore
            from scapy.utils import rdpcap  # type: ignore
        except ImportError:
            raise ImportError(
                "scapy is required for pcap loading. Install it with: pip install scapy"
            ) from None

        try:
            packets = rdpcap(path)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to read pcap file {path!r}: {exc}") from exc

        count = 0
        parsed = 0
        failed = 0
        for pkt in packets:
            count += 1
            # Guarded per-packet parse. Everything from the link-layer probe
            # (including the IP-layer probe below) through the final yield is
            # wrapped so a single bad frame yields None (matching
            # PacketParser.from_raw's None for unparseable input) and the loader
            # keeps processing the rest of the capture. This also prevents a
            # partial layer from leaking None-valued fields into the int() casts
            # or from aborting the whole async generator.
            try:
                ip_layer = self._parse_ip(pkt)
                if ip_layer is None:
                    # Non-IPv4 frame (e.g. ARP) or unsupported link — unparseable,
                    # mirroring PacketParser.from_raw returning None for the same.
                    yield None
                    continue

                # Recover L2 addresses and detect the link-layer header size so
                # we can mirror PacketParser.from_raw's `len(data) < 20`
                # minimum-IPv4 guard on the captured byte length (minus link).
                src_mac = dst_mac = ""
                eth = pkt.getlayer(Ether)
                if eth is not None:
                    # Ethernet base header is 14 bytes. Each stacked 802.1Q VLAN
                    # tag adds 4 bytes: one tag -> 18, QinQ/double-tag -> 22, etc.
                    # Walk the Dot1Q chain by descending through each tag's
                    # payload (Dot1Q.getlayer("Dot1Q") returns the first Dot1Q
                    # layer, which for a Dot1Q packet is itself — so re-querying
                    # it would loop forever). We advance via the layer's payload
                    # until it is no longer a Dot1Q.
                    link_offset = 14
                    layer = eth.payload
                    while isinstance(layer, Dot1Q):
                        link_offset += 4
                        layer = layer.payload
                    src_mac = eth.src if hasattr(eth, "src") else ""
                    dst_mac = eth.dst if hasattr(eth, "dst") else ""
                elif pkt.haslayer("cooked linux") or pkt.haslayer("LinuxSLL"):
                    link_offset = 16  # SLL v1 (classic tcpdump -i any) 16-byte header
                elif pkt.haslayer("cooked linux v2") or pkt.haslayer("LinuxSLL2"):
                    link_offset = 20  # SLL2 (modern kernels/libpcap tcpdump -i any) 20-byte header
                else:
                    link_offset = 0

                # Mirror PacketParser.from_raw's guards one for one, so the two
                # ingestion paths accept and reject the same frames.
                version = int(ip_layer.version)
                ihl = int(ip_layer.ihl) * 4
                ip_total = int(ip_layer.len)
                avail = len(bytes(pkt)) - link_offset
                if (version != 4
                        or avail < 20
                        or ihl < 20 or ihl > 60 or ihl > avail
                        or ip_total < ihl
                        # A non-first fragment carries no L4 header, so reading
                        # ports at `ihl` yields arbitrary payload bytes.
                        or int(ip_layer.frag) != 0):
                    yield None
                    continue

                tcp_layer = ip_layer.getlayer(TCP)
                udp_layer = ip_layer.getlayer(UDP)
                proto = int(ip_layer.proto)

                src_port = dst_port = tcp_flags = window_size = 0
                if proto == 6:  # TCP
                    if tcp_layer is None or len(bytes(tcp_layer)) < 20:
                        yield None
                        continue
                    l4_header = int(tcp_layer.dataofs) * 4
                    if (l4_header < 20 or l4_header > 60
                            or avail < ihl + l4_header
                            or ip_total < ihl + l4_header):
                        yield None
                        continue
                    src_port = int(tcp_layer.sport)
                    dst_port = int(tcp_layer.dport)
                    tcp_flags = int(tcp_layer.flags) & 0x3F
                    window_size = int(tcp_layer.window)
                elif proto == 17:  # UDP
                    if udp_layer is None or len(bytes(udp_layer)) < 8:
                        yield None
                        continue
                    l4_header = 8
                    if avail < ihl + l4_header or ip_total < ihl + l4_header:
                        yield None
                        continue
                    src_port = int(udp_layer.sport)
                    dst_port = int(udp_layer.dport)
                else:
                    # Any other protocol still yields a record with ports at
                    # zero, exactly like from_raw: the rule engine's
                    # allowed_protocols filter blocks it, which is a more
                    # auditable outcome than a silent parse failure.  No L4
                    # header is subtracted, matching from_raw (the old code
                    # subtracted the UDP default of 8 here, so ICMP
                    # payload_size drifted from the live path).
                    l4_header = 0

                payload_size = max(0, ip_total - ihl - l4_header)

                yield {
                    "src_ip": ip_layer.src,
                    "dst_ip": ip_layer.dst,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    # protocol is taken from the IP layer so it stays correct
                    # even when only one of TCP/UDP is present.
                    "protocol": proto,
                    # IP-layer packet size (matches PacketParser.from_raw's
                    # total_len), kept distinct from `payload_size` which is
                    # the L4 payload.
                    "packet_size": ip_total,
                    "timestamp": float(pkt.time),
                    "src_mac": src_mac,
                    "dst_mac": dst_mac,
                    "tcp_flags": tcp_flags,
                    "ttl": int(ip_layer.ttl),
                    "payload_size": payload_size,
                    "window_size": window_size,
                }
                parsed += 1
            except Exception:  # noqa: BLE001
                # Malformed IPv4/L4 frame — treat as unparseable (matching
                # PacketParser.from_raw returning None for the same) and keep
                # processing the rest of the capture.
                failed += 1
                yield None

        logger.info(
            "Loaded %d packets from %s (%d parsed as IPv4, %d unparseable)",
            count, path, parsed, failed,
        )

    # -- link-layer parsing --------------------------------------------------

    @staticmethod
    def _parse_ip(pkt):
        """Return the IP layer for a scapy packet, or None if not IPv4-parseable.

        scapy is imported lazily here (matching load()'s deferred import) —
        ruff >= 0.16 no longer accepts the previous globals().update() binding
        trick for these names (F821), and a function-local import makes the
        dependency explicit at the only site that needs it.
        """
        try:
            from scapy.layers.inet import IP  # type: ignore
        except ImportError:
            raise ImportError(
                "scapy is required for pcap loading. Install it with: pip install scapy"
            ) from None
        try:
            ip_layer = pkt.getlayer(IP)
            return ip_layer
        except Exception:  # noqa: BLE001
            return None
