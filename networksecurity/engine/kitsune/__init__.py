"""
Kitsune — Online network intrusion detection via incremental statistics and autoencoder ensemble.
Reference: ymirsky/Kitsune-py (NDSS'18)

Core components:
- AfterImage: damped incremental statistics for 115-dim per-packet features
- KitNET: autoencoder ensemble for unsupervised anomaly detection
"""

from networksecurity.engine.kitsune.afterimage import AfterImage, IncStat, IncStatDB
from networksecurity.engine.kitsune.kitnet import AutoEncoder, KitNET
from networksecurity.engine.kitsune.kitsune import Kitsune

__all__ = [
    'AfterImage',
    'AutoEncoder',
    'IncStat',
    'IncStatDB',
    'KitNET',
    'Kitsune',
]
