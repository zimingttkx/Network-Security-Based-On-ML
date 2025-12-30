"""
LUCID - 轻量级CNN DDoS检测
参考: doriguzzi/lucid-ddos

LUCID使用CNN学习DDoS和正常流量的行为模式，
具有低处理开销和快速检测时间。
"""

from networksecurity.models.lucid.cnn import LucidCNN
from networksecurity.models.lucid.dataset_parser import LucidDatasetParser, FlowSample
from networksecurity.models.lucid.detector import LucidDetector

__all__ = [
    'LucidCNN',
    'LucidDatasetParser', 
    'FlowSample',
    'LucidDetector'
]
