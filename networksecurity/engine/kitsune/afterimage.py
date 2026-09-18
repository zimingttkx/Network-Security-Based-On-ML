"""
AfterImage — Damped incremental statistics engine for network traffic.
Based on ymirsky/Kitsune-py (NDSS'18).

Tracks temporal patterns across 5 exponentially-decaying time windows
(5s, 3s, 1s, 0.1s, 0.01s) over 5 traffic channels (MAC pair, IP src,
IP pair, socket src, socket pair), producing 90 features per packet.
"""

import logging
from collections import OrderedDict
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)

# MAC values meaning "the capture source has no link-layer header to give us"
# (the live NFQUEUE path parses at the IP layer).
_ABSENT_MAC: frozenset[str] = frozenset(("", "00:00:00:00:00:00"))


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
    
    def insert(self, value: float, timestamp: float = 0.0, src_value: float | None = None):
        """Insert a new value and update statistics with time decay."""
        # Apply time decay
        if timestamp < self.last_timestamp:
            # Clock stepped backwards (NTP correction, or a wall-clock /
            # monotonic mix-up upstream).  Decaying by a negative delta would
            # multiply the accumulated weight by exp(+lambda*|dt|) and blow the
            # statistics up; leaving last_timestamp pinned in the future
            # instead freezes the window until the clock catches up.  Rebase
            # onto the earlier timestamp and keep the weights as they are.
            self.last_timestamp = timestamp
        elif timestamp > self.last_timestamp:
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

    def get_stats(self) -> tuple[float, float, float]:
        """Return (weight, mean, std)."""
        return self.weight, self.mean(), self.std()


class IncStatDB:
    """Keyed database of incremental statistics.

    Bounded by ``max_hosts``: once the cache exceeds that size, the
    least-recently-used entry is evicted (LRU).  This keeps memory
    bounded during long-running live interception.
    """

    def __init__(self, lambda_: float = 1.0, max_hosts: int = 10000):
        self.lambda_ = lambda_
        self.max_hosts = max(1, max_hosts)
        self.stats: "OrderedDict[str, IncStat]" = OrderedDict()

    def get_stat(self, key: str, init_time: float = 0.0) -> IncStat:
        """Get or create an IncStat for `key`."""
        stat = self.stats.get(key)
        if stat is None:
            if len(self.stats) >= self.max_hosts:
                # Evict the least-recently-used entry.
                self.stats.popitem(last=False)
            stat = IncStat(self.lambda_, init_time)
            self.stats[key] = stat
        else:
            # Mark as recently used.
            self.stats.move_to_end(key)
        return stat

    def update(self, key: str, value: float, timestamp: float = 0.0):
        """Update the statistic for `key` with a new value."""
        stat = self.get_stat(key, timestamp)
        stat.insert(value, timestamp)

    def get_stats(self, key: str) -> tuple[float, float, float]:
        """Return (weight, mean, std) for `key`."""
        stat = self.stats.get(key)
        if stat is not None:
            return stat.get_stats()
        return 0.0, 0.0, 0.0


class AfterImage:
    """
    AfterImage feature extractor.

    Extracts 90 features per packet across five traffic channels:
    - MAC pair (18 features)
    - IP source (18 features)
    - IP pair (18 features)
    - Socket source (18 features)
    - Socket pair (18 features)

    Each channel is tracked across 5 decay windows (5s, 3s, 1s, 0.1s, 0.01s),
    contributing [weight, mean, std] per window (15) plus 3 cross-window
    aggregates of the weights.
    """

    LAMBDAS: ClassVar[list] = [5, 3, 1, 0.1, 0.01]
    FEATURES_PER_CHANNEL: ClassVar[int] = len(LAMBDAS) * 3 + 3
    N_CHANNELS: ClassVar[int] = 5
    FEATURE_DIM: ClassVar[int] = FEATURES_PER_CHANNEL * N_CHANNELS

    def __init__(self, max_hosts: int = 10000):
        """
        Args:
            max_hosts: Maximum number of hosts to track *per channel per
                time window*.  There are 5 channels × 5 windows = 25
                independent IncStatDBs, so the real upper bound on tracked
                IncStat objects is ``25 * max_hosts``.  The default (10k)
                keeps that at ~250k objects, which is bounded for edge
                deployment; raise it only if you have the RAM to spare.
        """
        self.max_hosts = max(1, max_hosts)

        # One IncStatDB per time window, per channel
        self.mac_stats = [IncStatDB(1.0 / l, max_hosts) for l in self.LAMBDAS]
        self.ip_stats = [IncStatDB(1.0 / l, max_hosts) for l in self.LAMBDAS]
        self.ip_pair_stats = [IncStatDB(1.0 / l, max_hosts) for l in self.LAMBDAS]
        self.socket_stats = [IncStatDB(1.0 / l, max_hosts) for l in self.LAMBDAS]
        self.socket_pair_stats = [IncStatDB(1.0 / l, max_hosts) for l in self.LAMBDAS]

        self.packet_count = 0

    def update_get_stats(self, src_mac: str, dst_mac: str, src_ip: str, dst_ip: str,
                         src_port: int, dst_port: int, packet_size: int,
                         timestamp: float, protocol: int = 0,
                         ttl: int = 0) -> np.ndarray:
        """Update all statistics and return the 90-dim feature vector.

        Args:
            src_mac: Source MAC address ("" when the capture source has none).
            dst_mac: Destination MAC address ("" when the capture source has none).
            src_ip: Source IP address.
            dst_ip: Destination IP address.
            src_port: Source port.
            dst_port: Destination port.
            packet_size: IP packet size in bytes.
            timestamp: Unix timestamp in seconds.
            protocol: IP protocol number; MAC-pair proxy key when MACs are absent.
            ttl: IP TTL; MAC-pair proxy key when MACs are absent.

        Returns:
            90-dimensional float32 feature vector.
        """
        self.packet_count += 1
        features = []

        # MAC pair channel (18 features)
        mac_key = self._mac_pair_key(src_mac, dst_mac, protocol, ttl)
        features.extend(self._extract_channel_features(
            self.mac_stats, mac_key, packet_size, timestamp))

        # IP source channel (18 features)
        features.extend(self._extract_channel_features(
            self.ip_stats, src_ip, packet_size, timestamp))

        # IP pair channel (18 features)
        ip_pair_key = f"{src_ip}->{dst_ip}"
        features.extend(self._extract_channel_features(
            self.ip_pair_stats, ip_pair_key, packet_size, timestamp))

        # Socket source channel (18 features)
        socket_src_key = f"{src_ip}:{src_port}"
        features.extend(self._extract_channel_features(
            self.socket_stats, socket_src_key, packet_size, timestamp))

        # Socket pair channel (18 features)
        socket_pair_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        features.extend(self._extract_channel_features(
            self.socket_pair_stats, socket_pair_key, packet_size, timestamp))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _mac_pair_key(src_mac: str, dst_mac: str, protocol: int, ttl: int) -> str:
        """Channel key for the MAC-pair statistics.

        The live NFQUEUE path hands us packets at the IP layer, so both MACs
        are empty and the naive key would be "->" for every packet — one
        global stream instead of a per-link channel, which also makes the
        feature map correlate the MAC channel with nothing.  Fall back to a
        (protocol, ttl) proxy so the channel still separates traffic classes.
        """
        if src_mac in _ABSENT_MAC and dst_mac in _ABSENT_MAC:
            return f"p{protocol}-t{ttl}"
        return f"{src_mac}->{dst_mac}"

    def _extract_channel_features(self, stat_dbs: list[IncStatDB], key: str,
                                   value: float, timestamp: float) -> list[float]:
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

        # Two things deliberately NOT appended here:
        #
        # - Normalized wall-clock (timestamp % 86400/3600/60), which an earlier
        #   revision emitted.  It is non-stationary — over any realistic
        #   training window the day/hour fractions are monotonic ramps, so
        #   detection-time values always sit outside the training range and
        #   the model flags EVERY new packet as anomalous.  Measured on a 55s
        #   training window: FPR 0.4% at t=0 grows to 100% within 10 minutes of
        #   wall-clock drift and stays there for 12h (RMSE 0.44 -> 8.9 -> 328).
        #   Temporal dynamics are already captured by the damped decay windows
        #   above; wall-clock has no place in per-packet anomaly features (the
        #   Kitsune paper's AfterImage has none either).
        #
        # - A [log1p(packet_size), 1.0] pair.  packet_size is the value fed
        #   into every window, so the per-window means already carry it; the
        #   log1p term duplicated the 0.01s window and the 1.0 was a constant
        #   contributing no variance while still claiming a slot in the
        #   feature map.  Dropping the pair takes the vector from 100 to 90
        #   dimensions.

        return features

    def get_feature_dim(self) -> int:
        """Return the feature dimension (90)."""
        return self.FEATURE_DIM

    def reset(self):
        """Reset all statistics."""
        self.mac_stats = [IncStatDB(1.0 / l, self.max_hosts) for l in self.LAMBDAS]
        self.ip_stats = [IncStatDB(1.0 / l, self.max_hosts) for l in self.LAMBDAS]
        self.ip_pair_stats = [IncStatDB(1.0 / l, self.max_hosts) for l in self.LAMBDAS]
        self.socket_stats = [IncStatDB(1.0 / l, self.max_hosts) for l in self.LAMBDAS]
        self.socket_pair_stats = [IncStatDB(1.0 / l, self.max_hosts) for l in self.LAMBDAS]
        self.packet_count = 0
