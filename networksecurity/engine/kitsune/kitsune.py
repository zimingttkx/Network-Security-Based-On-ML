"""
Kitsune — Online, unsupervised network intrusion detection system.
Based on ymirsky/Kitsune-py (NDSS'18).

Uses AfterImage for incremental statistical feature extraction
and KitNET autoencoder ensemble for anomaly detection.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from networksecurity.engine.kitsune.afterimage import AfterImage
from networksecurity.engine.kitsune.kitnet import KitNET

logger = logging.getLogger(__name__)


@dataclass
class KitsuneResult:
    """Kitsune detection result."""
    rmse: float
    is_anomaly: bool
    packet_count: int
    is_training: bool
    threshold: float | None = None
    
    def to_dict(self) -> dict:
        return {
            'rmse': self.rmse,
            'is_anomaly': self.is_anomaly,
            'packet_count': self.packet_count,
            'is_training': self.is_training,
            'threshold': self.threshold
        }


class Kitsune:
    """
    Kitsune network intrusion detection system.

    - Online: per-packet processing, no history storage.
    - Unsupervised: trains on normal traffic, no labels needed.
    - Efficient: lightweight autoencoders for edge deployment.

    Usage:
    ```python
    kitsune = Kitsune()
    for packet in packets:
        result = kitsune.process(packet)
        if result.is_anomaly:
            print(f"Anomaly detected! RMSE={result.rmse}")
    ```
    """
    
    def __init__(self,
                 max_autoencoder_size: int = 10,
                 fm_grace_period: int = 5000,
                 ad_grace_period: int = 50000,
                 learning_rate: float = 0.1,
                 threshold_percentile: float = 99.0):
        self.max_ae_size = max_autoencoder_size
        self.fm_grace = fm_grace_period
        self.ad_grace = ad_grace_period
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile

        # Components
        self.afterimage = AfterImage()
        self.kitnet: KitNET | None = None

        # State
        self.packet_count = 0
        self.is_initialized = False
        self._start_time = time.time()
    
    def _initialize_kitnet(self, feature_dim: int):
        """Initialize KitNET"""
        self.kitnet = KitNET(
            input_dim=feature_dim,
            max_autoencoder_size=self.max_ae_size,
            fm_grace_period=self.fm_grace,
            ad_grace_period=self.ad_grace,
            learning_rate=self.learning_rate,
            threshold_percentile=self.threshold_percentile,
        )
        self.is_initialized = True
        logger.info("Kitsune: KitNET initialized, feature_dim=%d", feature_dim)

    def set_grace_periods(self, fm_grace_period: int | None = None,
                          ad_grace_period: int | None = None) -> None:
        """Override the feature-mapping / anomaly-detection grace periods.

        Must be called before any packet is processed (KitNET is not yet
        initialized at that point).  Useful for benchmarks that need shorter
        training windows than the production defaults.
        """
        if self.is_initialized:
            raise RuntimeError("grace periods can only be set before training starts")
        if fm_grace_period is not None:
            self.fm_grace = fm_grace_period
        if ad_grace_period is not None:
            self.ad_grace = ad_grace_period
    
    def process_packet(self, packet_info: dict) -> KitsuneResult:
        """Process a single packet.  Accepts a dict with:
        src_mac, dst_mac, src_ip, dst_ip, src_port, dst_port,
        packet_size, timestamp.
        """
        self.packet_count += 1

        features = self.afterimage.update_get_stats(
            src_mac=packet_info.get("src_mac", ""),
            dst_mac=packet_info.get("dst_mac", ""),
            src_ip=packet_info.get("src_ip", "0.0.0.0"),
            dst_ip=packet_info.get("dst_ip", "0.0.0.0"),
            src_port=packet_info.get("src_port", 0),
            dst_port=packet_info.get("dst_port", 0),
            packet_size=packet_info.get("packet_size", 0),
            timestamp=packet_info.get("timestamp", 0.0),
        )

        return self._process_features(features)

    def _process_features(self, features: np.ndarray) -> KitsuneResult:
        """Process a feature vector through KitNET."""
        if not self.is_initialized:
            self._initialize_kitnet(len(features))

        rmse = self.kitnet.process(features)
        is_training = not self.kitnet.is_ad_done
        is_anomaly = self.kitnet.is_anomaly(rmse) if not is_training else False

        return KitsuneResult(
            rmse=rmse,
            is_anomaly=is_anomaly,
            packet_count=self.packet_count,
            is_training=is_training,
            threshold=self.kitnet.threshold
        )

    def get_state(self) -> dict:
        """Get model state."""
        state = {
            'packet_count': self.packet_count,
            'is_initialized': self.is_initialized,
            'runtime_seconds': time.time() - self._start_time
        }
        if self.kitnet:
            state.update(self.kitnet.get_state())
        return state
    
    def reset(self):
        """Reset the model."""
        self.afterimage.reset()
        self.kitnet = None
        self.packet_count = 0
        self.is_initialized = False
        self._start_time = time.time()
    
    @property
    def is_ready(self) -> bool:
        """Check whether training is complete."""
        return self.kitnet is not None and self.kitnet.is_ad_done
