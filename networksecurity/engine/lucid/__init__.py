"""
LUCID — Lightweight CNN DDoS detection.
Reference: doriguzzi/lucid-ddos (IEEE TNSM 2020)

LUCID uses a 1D CNN to learn spatiotemporal patterns from network flows,
delivering low-overhead, real-time DDoS detection.
"""

from networksecurity.engine.lucid.cnn import LucidCNN
from networksecurity.engine.lucid.dataset_parser import LucidDatasetParser, FlowSample
from networksecurity.engine.lucid.detector import LucidDetector

__all__ = [
    'LucidCNN',
    'LucidDatasetParser', 
    'FlowSample',
    'LucidDetector'
]
