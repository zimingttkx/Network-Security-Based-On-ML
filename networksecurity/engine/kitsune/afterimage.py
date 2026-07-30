"""
AfterImage — Damped incremental statistics engine for network traffic.
Based on ymirsky/Kitsune-py (NDSS'18).

Tracks temporal patterns across 5 exponentially-decaying time windows
(5s, 3s, 1s, 0.1s, 0.01s) over 5 traffic channels (MAC pair, IP src,
IP pair, socket src, socket pair), producing 115 features per packet.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IncStat:
    """Incremental statistics tracker with exponential decay."""
    
    def __init__(self, lambda_: float = 1.0, init_time: float = 0.0, is_typed: bool = False):
        """
        Args:
            lambda_: Decay factor controlling weight of historical data.
            init_time: Initial timestamp.
            is_typed: Whether to group by type.
        """
        self.lambda_ = lambda_
        self.is_typed = is_typed
        self.last_timestamp = init_time
        
        # First-order statistic (mean)
        self.weight = 0.0
        self.sum = 0.0

        # Second-order statistic (variance)
        self.sum_sq = 0.0

        # Covariance-related
        self.src_sum = 0.0
        self.src_sum_sq = 0.0
        self.cov_sum = 0.0
    
    def insert(self, value: float, timestamp: float = 0.0, src_value: float = None):
        """Insert a new value and update statistics with time decay."""
        # Apply time decay
        if timestamp > self.last_timestamp:
            time_diff = timestamp - self.last_timestamp
            decay = np.exp(-self.lambda_ * time_diff)
            self.weight *= decay
            self.sum *= decay
            self.sum_sq *= decay
            if src_value is not None:
                self.src_sum *= decay
                self.src_sum_sq *= decay
                self.cov_sum *= decay
            self.last_timestamp = timestamp
        
        # Update statistics
        self.weight += 1.0
        self.sum += value
        self.sum_sq += value * value
        
        if src_value is not None:
            self.src_sum += src_value
            self.src_sum_sq += src_value * src_value
            self.cov_sum += value * src_value
    
    def mean(self) -> float:
        """Weighted mean."""
        if self.weight < 1e-10:
            return 0.0
        return self.sum / self.weight
    
    def var(self) -> float:
        """Weighted variance."""
        if self.weight < 2:
            return 0.0
        mean = self.mean()
        return max(0, self.sum_sq / self.weight - mean * mean)

    def std(self) -> float:
        """Weighted standard deviation."""
        return np.sqrt(self.var())

    def cov(self) -> float:
        """Weighted covariance."""
        if self.weight < 2:
            return 0.0
        return self.cov_sum / self.weight - (self.sum / self.weight) * (self.src_sum / self.weight)

    def pcc(self) -> float:
        """Pearson correlation coefficient."""
        if self.weight < 2:
            return 0.0

        var_x = self.var()
        var_y = max(0, self.src_sum_sq / self.weight - (self.src_sum / self.weight) ** 2)

        if var_x < 1e-10 or var_y < 1e-10:
            return 0.0

        return self.cov() / (np.sqrt(var_x) * np.sqrt(var_y))

    def get_stats(self) -> Tuple[float, float, float]:
        """Return (weight, mean, std)."""
        return self.weight, self.mean(), self.std()

    def get_stats_1d(self) -> List[float]:
        """Return 1-d stats: [weight, mean, std]."""
        return [self.weight, self.mean(), self.std()]

    def get_stats_2d(self) -> List[float]:
        """Return 2-d stats: [weight, mean, std, cov, pcc]."""
        return [self.weight, self.mean(), self.std(), self.cov(), self.pcc()]


class IncStatDB:
    """Keyed database of incremental statistics."""

    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.stats: Dict[str, IncStat] = {}

    def get_stat(self, key: str, init_time: float = 0.0) -> IncStat:
        """Get or create an IncStat for `key`."""
        if key not in self.stats:
            self.stats[key] = IncStat(self.lambda_, init_time)
        return self.stats[key]

    def update(self, key: str, value: float, timestamp: float = 0.0):
        """Update the statistic for `key` with a new value."""
        stat = self.get_stat(key, timestamp)
        stat.insert(value, timestamp)

    def get_stats(self, key: str) -> Tuple[float, float, float]:
        """Return (weight, mean, std) for `key`."""
        if key in self.stats:
            return self.stats[key].get_stats()
        return 0.0, 0.0, 0.0


class AfterImage:
    """
    AfterImage feature extractor.

    Extracts 115 features per packet across five traffic channels:
    - MAC pair (23 features)
    - IP source (23 features)
    - IP pair (23 features)
    - Socket source (23 features)
    - Socket pair (23 features)

    Each channel is tracked across 5 decay windows (5s, 3s, 1s, 0.1s, 0.01s).
    """

    LAMBDAS = [5, 3, 1, 0.1, 0.01]

    def __init__(self, max_hosts: int = 100000):
        """
        Args:
            max_hosts: Maximum number of hosts to track simultaneously.
        """
        self.max_hosts = max_hosts

        # One IncStatDB per time window, per channel
        self.mac_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]

        self.packet_count = 0

    def update_get_stats(self, src_mac: str, dst_mac: str, src_ip: str, dst_ip: str,
                         src_port: int, dst_port: int, packet_size: int,
                         timestamp: float) -> np.ndarray:
        """Update all statistics and return the 115-dim feature vector.

        Args:
            src_mac: Source MAC address.
            dst_mac: Destination MAC address.
            src_ip: Source IP address.
            dst_ip: Destination IP address.
            src_port: Source port.
            dst_port: Destination port.
            packet_size: IP packet size in bytes.
            timestamp: Unix timestamp in seconds.

        Returns:
            115-dimensional float32 feature vector.
        """
        self.packet_count += 1
        features = []

        # MAC pair channel (23 features)
        mac_key = f"{src_mac}->{dst_mac}"
        features.extend(self._extract_channel_features(
            self.mac_stats, mac_key, packet_size, timestamp))

        # IP source channel (23 features)
        features.extend(self._extract_channel_features(
            self.ip_stats, src_ip, packet_size, timestamp))

        # IP pair channel (23 features)
        ip_pair_key = f"{src_ip}->{dst_ip}"
        features.extend(self._extract_channel_features(
            self.ip_pair_stats, ip_pair_key, packet_size, timestamp))

        # Socket source channel (23 features)
        socket_src_key = f"{src_ip}:{src_port}"
        features.extend(self._extract_channel_features(
            self.socket_stats, socket_src_key, packet_size, timestamp))

        # Socket pair channel (23 features)
        socket_pair_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        features.extend(self._extract_channel_features(
            self.socket_pair_stats, socket_pair_key, packet_size, timestamp))

        return np.array(features, dtype=np.float32)

    def _extract_channel_features(self, stat_dbs: List[IncStatDB], key: str,
                                   value: float, timestamp: float) -> List[float]:
        """Extract features for a single channel across all time windows."""
        features = []

        for db in stat_dbs:
            db.update(key, value, timestamp)
            weight, mean, std = db.get_stats(key)
            features.extend([weight, mean, std])

        # Aggregate statistics across windows
        all_weights = [db.get_stats(key)[0] for db in stat_dbs]
        features.extend([
            np.mean(all_weights),
            np.std(all_weights),
            np.max(all_weights) - np.min(all_weights) if all_weights else 0,
        ])

        # Time and size features
        features.extend([
            np.log1p(value),
            timestamp % 86400 / 86400,   # time-of-day normalized
            timestamp % 3600 / 3600,      # hour-of-day normalized
            timestamp % 60 / 60,          # minute-of-hour normalized
            1.0,                          # placeholder
        ])

        return features

    def extract_features_from_packet(self, packet_info: Dict) -> np.ndarray:
        """Extract features from a packet info dict.

        Args:
            packet_info: dict with keys src_mac, dst_mac, src_ip, dst_ip,
                         src_port, dst_port, packet_size, timestamp.
        """
        return self.update_get_stats(
            src_mac=packet_info.get('src_mac', '00:00:00:00:00:00'),
            dst_mac=packet_info.get('dst_mac', '00:00:00:00:00:00'),
            src_ip=packet_info.get('src_ip', '0.0.0.0'),
            dst_ip=packet_info.get('dst_ip', '0.0.0.0'),
            src_port=packet_info.get('src_port', 0),
            dst_port=packet_info.get('dst_port', 0),
            packet_size=packet_info.get('packet_size', 0),
            timestamp=packet_info.get('timestamp', 0.0),
        )

    def get_feature_dim(self) -> int:
        """Return the feature dimension (115)."""
        return 115

    def reset(self):
        """Reset all statistics."""
        self.mac_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.packet_count = 0
