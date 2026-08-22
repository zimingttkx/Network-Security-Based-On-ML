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
    label: int = 0  # 0=normal, 1=DDoS
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    
    def add_packet(self, packet: dict):
        """Add a packet to this flow."""
        self.packets.append(packet)
        if not self.timestamp_start:
            self.timestamp_start = packet.get('timestamp', 0)
        self.timestamp_end = packet.get('timestamp', 0)
    
    @property
    def duration(self) -> float:
        return self.timestamp_end - self.timestamp_start
    
    @property
    def packet_count(self) -> int:
        return len(self.packets)


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
        """Check whether traffic is from an attacker."""
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        return src_ip in self.attacker_ips or dst_ip in self.victim_ips
    
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
        
        # TCP flags
        features[3] = packet.get('tcp_flags', 0) / 255.0
        
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
            (feature_matrix, label) if flow complete, else None.
        """
        flow_id = self._get_flow_id(packet)
        
        # Get or create flow
        if flow_id not in self.flows:
            flow = FlowSample(
                flow_id=flow_id,
                label=1 if self._is_attack(packet) else 0
            )
            self._register_flow(flow_id, flow)
        else:
            flow = self.flows[flow_id]
        flow.add_packet(packet)
        
        # Check if sample is complete
        if flow.packet_count >= self.packets_per_flow:
            sample = self._create_sample(flow)
            del self.flows[flow_id]
            return sample, flow.label
        
        # Check time window
        if flow.duration >= self.time_window and flow.packet_count > 0:
            sample = self._create_sample(flow)
            del self.flows[flow_id]
            return sample, flow.label
        
        return None
    
    def _create_sample(self, flow: FlowSample) -> np.ndarray:
        """Create sample matrix."""
        sample = np.zeros((self.packets_per_flow, self.n_features), dtype=np.float32)
        
        prev_timestamp = 0
        for i, packet in enumerate(flow.packets[:self.packets_per_flow]):
            sample[i] = self._extract_packet_features(packet, prev_timestamp)
            prev_timestamp = packet.get('timestamp', 0)
        
        return sample
    
    def flush_flows(self) -> list[tuple[np.ndarray, int]]:
        """Flush all incomplete flows."""
        samples = []
        for flow in self.flows.values():
            if flow.packet_count > 0:
                sample = self._create_sample(flow)
                samples.append((sample, flow.label))
        self.flows.clear()
        return samples
    
    def parse_batch(self, packets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse a batch of packets.

        Returns:
            (X, y) feature matrix and labels.
        """
        samples = []
        labels = []
        
        for packet in packets:
            result = self.process_packet(packet)
            if result:
                samples.append(result[0])
                labels.append(result[1])
        
        # Flush remaining flows
        for sample, label in self.flush_flows():
            samples.append(sample)
            labels.append(label)
        
        if not samples:
            return np.array([]).reshape(0, self.packets_per_flow, self.n_features), np.array([])
        
        return np.array(samples), np.array(labels)
    
    def get_input_shape(self) -> tuple[int, int]:
        """Get input shape."""
        return (self.packets_per_flow, self.n_features)
