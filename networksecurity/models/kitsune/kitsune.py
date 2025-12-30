"""
Kitsune - 完整的网络入侵检测系统
基于 ymirsky/Kitsune-py (NDSS'18)

Kitsune是一个在线、无监督、高效的网络入侵检测系统。
它使用AfterImage进行增量统计特征提取，使用KitNET进行异常检测。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time

from networksecurity.models.kitsune.afterimage import AfterImage
from networksecurity.models.kitsune.kitnet import KitNET
from networksecurity.models.kitsune.feature_extractor import KitsuneFeatureExtractor, PacketInfo

logger = logging.getLogger(__name__)


@dataclass
class KitsuneResult:
    """Kitsune检测结果"""
    rmse: float
    is_anomaly: bool
    packet_count: int
    is_training: bool
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'rmse': self.rmse,
            'is_anomaly': self.is_anomaly,
            'packet_count': self.packet_count,
            'is_training': self.is_training,
            'threshold': self.threshold
        }


class Kitsune:
    """
    Kitsune网络入侵检测系统
    
    特点:
    - 在线学习: 逐包处理，无需存储历史数据
    - 无监督: 只需正常流量训练，无需标签
    - 高效: 轻量级自编码器，适合边缘设备
    
    使用方法:
    ```python
    kitsune = Kitsune()
    for packet in packets:
        result = kitsune.process(packet)
        if result.is_anomaly:
            print(f"检测到异常! RMSE={result.rmse}")
    ```
    """
    
    def __init__(self, 
                 max_autoencoder_size: int = 10,
                 fm_grace_period: int = 5000,
                 ad_grace_period: int = 50000,
                 learning_rate: float = 0.1,
                 threshold_percentile: float = 99.0,
                 use_afterimage: bool = True):
        """
        Args:
            max_autoencoder_size: 单个自编码器最大输入维度
            fm_grace_period: 特征映射学习期
            ad_grace_period: 异常检测训练期
            learning_rate: 学习率
            threshold_percentile: 异常阈值百分位数
            use_afterimage: 是否使用AfterImage特征提取
        """
        self.max_ae_size = max_autoencoder_size
        self.fm_grace = fm_grace_period
        self.ad_grace = ad_grace_period
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.use_afterimage = use_afterimage
        
        # 组件
        self.afterimage = AfterImage() if use_afterimage else None
        self.feature_extractor = KitsuneFeatureExtractor()
        self.kitnet: Optional[KitNET] = None
        
        # 状态
        self.packet_count = 0
        self.is_initialized = False
        self._start_time = time.time()
    
    def _initialize_kitnet(self, feature_dim: int):
        """初始化KitNET"""
        self.kitnet = KitNET(
            input_dim=feature_dim,
            max_autoencoder_size=self.max_ae_size,
            fm_grace_period=self.fm_grace,
            ad_grace_period=self.ad_grace,
            learning_rate=self.learning_rate
        )
        self.is_initialized = True
        logger.info(f"Kitsune: 初始化KitNET，特征维度={feature_dim}")
    
    def process_packet(self, packet_info: Union[PacketInfo, Dict]) -> KitsuneResult:
        """
        处理单个数据包
        
        Args:
            packet_info: 数据包信息
            
        Returns:
            检测结果
        """
        self.packet_count += 1
        
        # 转换为PacketInfo
        if isinstance(packet_info, dict):
            packet = PacketInfo(**packet_info)
        else:
            packet = packet_info
        
        # 提取特征
        if self.use_afterimage and self.afterimage:
            features = self.afterimage.update_get_stats(
                src_mac=packet.src_mac,
                dst_mac=packet.dst_mac,
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                src_port=packet.src_port,
                dst_port=packet.dst_port,
                packet_size=packet.packet_size,
                timestamp=packet.timestamp
            )
        else:
            features = self.feature_extractor.extract(packet)
        
        return self._process_features(features)
    
    def process(self, data: Any) -> KitsuneResult:
        """
        处理任意格式的输入数据
        
        支持:
        - PacketInfo对象
        - 字典格式的数据包信息
        - numpy数组特征向量
        - 列表格式的特征
        """
        self.packet_count += 1
        
        if isinstance(data, (PacketInfo, dict)) and self.use_afterimage:
            return self.process_packet(data)
        
        # 直接处理特征向量
        features = self.feature_extractor.extract(data)
        return self._process_features(features)
    
    def _process_features(self, features: np.ndarray) -> KitsuneResult:
        """处理特征向量"""
        # 延迟初始化KitNET
        if not self.is_initialized:
            self._initialize_kitnet(len(features))
        
        # 处理
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
    
    def batch_process(self, data_list: List[Any]) -> List[KitsuneResult]:
        """批量处理"""
        return [self.process(d) for d in data_list]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        批量预测 (sklearn兼容接口)
        
        Returns:
            1 (正常) 或 -1 (异常)
        """
        results = []
        for x in X:
            result = self.process(x)
            results.append(-1 if result.is_anomaly else 1)
        return np.array(results)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        返回异常概率 (sklearn兼容接口)
        """
        probas = []
        for x in X:
            result = self.process(x)
            if self.kitnet and self.kitnet.threshold:
                # 基于RMSE和阈值计算概率
                prob = min(1.0, result.rmse / (self.kitnet.threshold * 2))
            else:
                prob = 0.5
            probas.append([1 - prob, prob])
        return np.array(probas)
    
    def get_state(self) -> Dict:
        """获取模型状态"""
        state = {
            'packet_count': self.packet_count,
            'is_initialized': self.is_initialized,
            'use_afterimage': self.use_afterimage,
            'runtime_seconds': time.time() - self._start_time
        }
        if self.kitnet:
            state.update(self.kitnet.get_state())
        return state
    
    def reset(self):
        """重置模型"""
        if self.afterimage:
            self.afterimage.reset()
        self.kitnet = None
        self.packet_count = 0
        self.is_initialized = False
        self._start_time = time.time()
    
    def is_ready(self) -> bool:
        """检查模型是否训练完成"""
        return self.kitnet is not None and self.kitnet.is_ad_done
