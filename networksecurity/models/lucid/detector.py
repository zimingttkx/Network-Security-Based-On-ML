"""
LUCID检测器
整合CNN模型和数据解析器的完整检测器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time

from networksecurity.models.lucid.cnn import LucidCNN
from networksecurity.models.lucid.dataset_parser import LucidDatasetParser, FlowSample

logger = logging.getLogger(__name__)


@dataclass
class LucidResult:
    """LUCID检测结果"""
    is_ddos: bool
    confidence: float
    flow_id: str = ""
    packets_analyzed: int = 0
    detection_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'is_ddos': self.is_ddos,
            'confidence': self.confidence,
            'flow_id': self.flow_id,
            'packets_analyzed': self.packets_analyzed,
            'detection_time_ms': self.detection_time_ms
        }


class LucidDetector:
    """
    LUCID DDoS检测器
    
    完整的DDoS检测流水线:
    1. 数据解析: 将原始数据包转换为流样本
    2. 特征提取: 提取时序特征
    3. CNN分类: 使用训练好的CNN进行分类
    
    使用方法:
    ```python
    detector = LucidDetector()
    detector.train(X_train, y_train)
    
    for packet in packets:
        result = detector.process_packet(packet)
        if result and result.is_ddos:
            print(f"DDoS攻击! 置信度={result.confidence}")
    ```
    """
    
    def __init__(self, time_window: float = 10.0, packets_per_flow: int = 10, **cnn_params):
        """
        Args:
            time_window: 时间窗口大小(秒)
            packets_per_flow: 每个样本的数据包数
            **cnn_params: CNN模型参数
        """
        self.time_window = time_window
        self.packets_per_flow = packets_per_flow
        
        # 数据解析器
        self.parser = LucidDatasetParser(time_window, packets_per_flow)
        
        # CNN模型
        cnn_params['time_steps'] = packets_per_flow
        cnn_params['n_features'] = self.parser.n_features
        self.cnn = LucidCNN(**cnn_params)
        
        # 状态
        self.is_trained = False
        self.total_packets = 0
        self.total_detections = 0
        self.ddos_detections = 0
    
    def set_attack_info(self, attackers: List[str], victims: List[str]):
        """设置攻击者和受害者IP (用于训练标签)"""
        self.parser.set_attack_info(attackers, victims)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, verbose: int = 0) -> Dict:
        """
        训练检测器
        
        Args:
            X: 训练数据 (n_samples, packets_per_flow, n_features)
            y: 标签 (0=正常, 1=DDoS)
        """
        result = self.cnn.fit(X, y, X_val, y_val, epochs=epochs, verbose=verbose)
        self.is_trained = True
        return result
    
    def train_from_packets(self, packets: List[Dict], epochs: int = 100, 
                           validation_split: float = 0.2, verbose: int = 0) -> Dict:
        """
        从原始数据包训练
        
        Args:
            packets: 数据包列表
            epochs: 训练轮数
            validation_split: 验证集比例
        """
        # 解析数据包
        X, y = self.parser.parse_batch(packets)
        
        if len(X) == 0:
            raise ValueError("没有足够的数据包生成训练样本")
        
        # 分割训练/验证集
        n_val = int(len(X) * validation_split)
        if n_val > 0:
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]
            X_train, X_val = X[n_val:], X[:n_val]
            y_train, y_val = y[n_val:], y[:n_val]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        return self.train(X_train, y_train, X_val, y_val, epochs, verbose)
    
    def process_packet(self, packet: Dict) -> Optional[LucidResult]:
        """
        处理单个数据包
        
        Returns:
            如果流完成，返回检测结果，否则返回None
        """
        self.total_packets += 1
        
        result = self.parser.process_packet(packet)
        if result is None:
            return None
        
        sample, _ = result
        return self._detect(sample)
    
    def _detect(self, sample: np.ndarray) -> LucidResult:
        """执行检测"""
        start_time = time.time()
        self.total_detections += 1
        
        if not self.is_trained:
            # 未训练时返回默认结果
            return LucidResult(
                is_ddos=False,
                confidence=0.0,
                packets_analyzed=self.packets_per_flow
            )
        
        # 预测
        sample_batch = sample.reshape(1, self.packets_per_flow, -1)
        proba = self.cnn.predict_proba(sample_batch)[0]
        is_ddos = proba[1] > 0.5
        
        if is_ddos:
            self.ddos_detections += 1
        
        detection_time = (time.time() - start_time) * 1000
        
        return LucidResult(
            is_ddos=is_ddos,
            confidence=float(proba[1]),
            packets_analyzed=self.packets_per_flow,
            detection_time_ms=detection_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """批量预测 (sklearn兼容接口)"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        return self.cnn.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        return self.cnn.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型"""
        return self.cnn.evaluate(X, y)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_packets': self.total_packets,
            'total_detections': self.total_detections,
            'ddos_detections': self.ddos_detections,
            'ddos_rate': self.ddos_detections / max(1, self.total_detections),
            'is_trained': self.is_trained
        }
    
    def reset_stats(self):
        """重置统计"""
        self.total_packets = 0
        self.total_detections = 0
        self.ddos_detections = 0
    
    def save(self, path: str):
        """保存模型"""
        self.cnn.save(path)
    
    def load(self, path: str):
        """加载模型"""
        self.cnn.load(path)
        self.is_trained = True
