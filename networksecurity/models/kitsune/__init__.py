"""
Kitsune - 基于增量统计和自编码器集成的网络入侵检测系统
参考: ymirsky/Kitsune-py (NDSS'18)

核心组件:
- AfterImage: 增量统计特征提取
- KitNET: 自编码器集成异常检测
"""

from networksecurity.models.kitsune.afterimage import AfterImage, IncStat, IncStatDB
from networksecurity.models.kitsune.kitnet import KitNET, AutoEncoder
from networksecurity.models.kitsune.feature_extractor import KitsuneFeatureExtractor
from networksecurity.models.kitsune.kitsune import Kitsune

__all__ = [
    'AfterImage',
    'IncStat', 
    'IncStatDB',
    'KitNET',
    'AutoEncoder',
    'KitsuneFeatureExtractor',
    'Kitsune'
]
