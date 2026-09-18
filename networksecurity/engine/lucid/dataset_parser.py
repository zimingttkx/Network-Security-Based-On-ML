"""
LUCID dataset parser.
Based on doriguzzi/lucid-ddos (IEEE TNSM 2020).

Converts raw network traffic into the input format required by the LUCID CNN.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FlowSample:
    """Flow sample."""
    flow_id: str
    packets: list[dict] = field(default_factory=list)
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    attack_votes: int = 0

    def add_packet(self, packet: dict, is_attack: bool = False):
        """Add a packet to this flow."""
        self.packets.append(packet)
        if is_attack:
            self.attack_votes += 1
        if not self.timestamp_start:
            self.timestamp_start = packet.get('timestamp', 0)
        self.timestamp_end = packet.get('timestamp', 0)

    @property
    def duration(self) -> float:
        return self.timestamp_end - self.timestamp_start

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def label(self) -> int:
        """Majority vote over the window's packets.

        The label used to be frozen from the first packet, so a flow that
        opened benignly and then carried the flood — or the reverse — was
        trained on whichever class its first packet happened to look like.
        """
        return 1 if self.attack_votes * 2 > len(self.packets) else 0


class LucidDatasetParser:
    """
    LUCID dataset parser.

    Converts raw traffic into CNN input:
    - Groups packets by flow (5-tuple).
    - Extracts per-packet features.
    - Produces fixed-size time-window samples.
    """
    
    # Per-packet feature names
    PACKET_FEATURES: ClassVar[list] = [
        'packet_size',      # packet size
        'iat',              # inter-arrival time
        'protocol',         # protocol (TCP=6, UDP=17)
        'tcp_flags',        # TCP flags
        'src_port_norm',    # normalized source port
        'dst_port_norm',    # normalized dest port
        'direction',        # direction (0=out, 1=in)
        'payload_size',     # payload size
        'header_size',      # header size
        'window_size',      # TCP window size
        'ttl'               # TTL value
    ]
    
    def __init__(self, time_window: float = 10.0, packets_per_flow: int = 10,
                 max_flows: int = 100000):
        """
        Args:
            time_window: time window in seconds.
            packets_per_flow: packets per flow sample.
            max_flows: maximum number of concurrent flows to track before
                       least-recently-created flows are evicted (bounds memory).
        """
        self.time_window = time_window
        self.packets_per_flow = packets_per_flow
        self.max_flows = max(1, max_flows)
        self.n_features = len(self.PACKET_FEATURES)

        # Flow buffer — bounded by max_flows (LRU eviction of oldest flows).
        self.flows: "OrderedDict[str, FlowSample]" = OrderedDict()

        # Windows discarded because they went stale before filling up.  Kept
        # as a counter rather than a log line: at line rate this fires often
        # and the number is what an operator needs, not each occurrence.
        self.expired_flows: int = 0

        # Attacker/victim IPs (for labeling)
        self.attacker_ips: set = set()
        self.victim_ips: set = set()

    def _register_flow(self, flow_id: str, flow: "FlowSample") -> None:
        """Insert a flow, evicting the oldest if over capacity."""
        if flow_id in self.flows:
            self.flows.move_to_end(flow_id)
        elif len(self.flows) >= self.max_flows:
            self.flows.popitem(last=False)  # evict oldest
        self.flows[flow_id] = flow
    
    def set_attack_info(self, attackers: list[str], victims: list[str]):
        """Set attacker and victim IPs."""
        self.attacker_ips = set(attackers)
        self.victim_ips = set(victims)
    
    def _get_flow_id(self, packet: dict) -> str:
        """Generate flow ID (5-tuple)."""
        src_ip = packet.get('src_ip', '0.0.0.0')
        dst_ip = packet.get('dst_ip', '0.0.0.0')
        src_port = packet.get('src_port', 0)
        dst_port = packet.get('dst_port', 0)
        protocol = packet.get('protocol', 6)
        
        # Bidirectional: sort to give both directions the same flow ID
        if (src_ip, src_port) > (dst_ip, dst_port):
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
    
    def _is_attack(self, packet: dict) -> bool:
        """Check whether a packet belongs to an attack, in either direction.

        Matching only ``src in attackers or dst in victims`` labelled the
        victim's own replies to a flood as normal, so a window carrying one
        event was split across both classes.
        """
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        return (src_ip in self.attacker_ips or dst_ip in self.attacker_ips
                or src_ip in self.victim_ips or dst_ip in self.victim_ips)

    def _sweep_expired(self, now: float) -> None:
        """Drop flows whose window elapsed without reaching a full sample.

        A silent flow used to sit in the buffer until LRU eviction, so a
        source that sent three packets, went quiet for ten minutes and came
        back completed a "window" spanning the whole gap — with every
        inter-arrival time clamped at the 1s cap.  Sweeping against the
        incoming packet's timestamp bounds the buffer by time as well as by
        count, and keeps a window's packets contemporaneous.

        Stale flows are discarded, never emitted: a window shorter than
        ``packets_per_flow`` has no real sample in it, and the padded
        all-zero rows the old code produced instead were fabricated traffic
        that entered training and inference alike.
        """
        if now <= 0 or not self.flows:
            return
        stale = [fid for fid, flow in self.flows.items()
                 if now - flow.timestamp_start >= self.time_window]
        for flow_id in stale:
            del self.flows[flow_id]
        self.expired_flows += len(stale)
    
    def _extract_packet_features(self, packet: dict, prev_timestamp: float = 0) -> np.ndarray:
        """Extract per-packet features."""
        features = np.zeros(self.n_features, dtype=np.float32)
        
        # Packet size (normalize to 0-1)
        features[0] = min(packet.get('packet_size', 0) / 1500.0, 1.0)
        
        # Inter-arrival time (normalize)
        timestamp = packet.get('timestamp', 0)
        iat = timestamp - prev_timestamp if prev_timestamp > 0 else 0
        features[1] = min(iat / 1.0, 1.0)  # cap at 1s
        
        # Protocol
        protocol = packet.get('protocol', 6)
        features[2] = 1.0 if protocol == 6 else (0.5 if protocol == 17 else 0.0)
        
        # TCP flags occupy the low 6 bits, so 63 is the real maximum —
        # dividing by 255 squeezed every combination into the bottom 2% of
        # the range and left the CNN nothing to separate.
        features[3] = min(packet.get('tcp_flags', 0) / 63.0, 1.0)
        
        # Ports (normalize)
        features[4] = packet.get('src_port', 0) / 65535.0
        features[5] = packet.get('dst_port', 0) / 65535.0
        
        # Direction
        features[6] = packet.get('direction', 0)
        
        # Payload and header size
        features[7] = min(packet.get('payload_size', 0) / 1500.0, 1.0)
        features[8] = min(packet.get('header_size', 20) / 60.0, 1.0)
        
        # Window size
        features[9] = min(packet.get('window_size', 0) / 65535.0, 1.0)
        
        # TTL
        features[10] = packet.get('ttl', 64) / 255.0
        
        return features
    
    def process_packet(self, packet: dict) -> tuple[np.ndarray, int] | None:
        """
        Process a single packet.

        Returns:
            (feature_matrix, label) when the flow's window filled up, else
            None.  A window only ever completes at exactly
            ``packets_per_flow`` packets; a flow that stalls is dropped by
            :meth:`_sweep_expired` instead of being padded out.
        """
        self._sweep_expired(packet.get('timestamp', 0))

        flow_id = self._get_flow_id(packet)
        flow = self.flows.get(flow_id)
        if flow is None:
            flow = FlowSample(flow_id=flow_id)
            self._register_flow(flow_id, flow)
        flow.add_packet(packet, self._is_attack(packet))

        if flow.packet_count >= self.packets_per_flow:
            del self.flows[flow_id]
            return self._create_sample(flow), flow.label

        return None
    
    def _create_sample(self, flow: FlowSample) -> np.ndarray:
        """Create sample matrix from a completed window."""
        sample = np.zeros((self.packets_per_flow, self.n_features), dtype=np.float32)
        
        prev_timestamp = 0
        for i, packet in enumerate(flow.packets[:self.packets_per_flow]):
            sample[i] = self._extract_packet_features(packet, prev_timestamp)
            prev_timestamp = packet.get('timestamp', 0)
        
        return sample
    
    def get_input_shape(self) -> tuple[int, int]:
        """Get input shape."""
        return (self.packets_per_flow, self.n_features)
