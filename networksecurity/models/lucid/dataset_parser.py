"""
LUCID数据集解析器
基于 doriguzzi/lucid-ddos 实现

将网络流量转换为LUCID CNN所需的输入格式。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FlowSample:
    """流量样本"""
    flow_id: str
    packets: List[Dict] = field(default_factory=list)
    label: int = 0  # 0=正常, 1=DDoS
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    
    def add_packet(self, packet: Dict):
        """添加数据包"""
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
    LUCID数据集解析器
    
    将原始网络流量转换为CNN输入格式:
    - 按流(5元组)分组数据包
    - 提取每个数据包的特征
    - 生成固定大小的时间窗口样本
    """
    
    # 每个数据包的特征
    PACKET_FEATURES = [
        'packet_size',      # 数据包大小
        'iat',              # 到达间隔时间
        'protocol',         # 协议 (TCP=6, UDP=17)
        'tcp_flags',        # TCP标志
        'src_port_norm',    # 归一化源端口
        'dst_port_norm',    # 归一化目标端口
        'direction',        # 方向 (0=出, 1=入)
        'payload_size',     # 负载大小
        'header_size',      # 头部大小
        'window_size',      # TCP窗口大小
        'ttl'               # TTL值
    ]
    
    def __init__(self, time_window: float = 10.0, packets_per_flow: int = 10):
        """
        Args:
            time_window: 时间窗口大小(秒)
            packets_per_flow: 每个样本的数据包数
        """
        self.time_window = time_window
        self.packets_per_flow = packets_per_flow
        self.n_features = len(self.PACKET_FEATURES)
        
        # 流量缓存
        self.flows: Dict[str, FlowSample] = {}
        
        # 攻击者/受害者IP (用于标签)
        self.attacker_ips: set = set()
        self.victim_ips: set = set()
    
    def set_attack_info(self, attackers: List[str], victims: List[str]):
        """设置攻击者和受害者IP"""
        self.attacker_ips = set(attackers)
        self.victim_ips = set(victims)
    
    def _get_flow_id(self, packet: Dict) -> str:
        """生成流ID (5元组)"""
        src_ip = packet.get('src_ip', '0.0.0.0')
        dst_ip = packet.get('dst_ip', '0.0.0.0')
        src_port = packet.get('src_port', 0)
        dst_port = packet.get('dst_port', 0)
        protocol = packet.get('protocol', 6)
        
        # 双向流: 排序确保同一连接的两个方向有相同ID
        if (src_ip, src_port) > (dst_ip, dst_port):
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
    
    def _is_attack(self, packet: Dict) -> bool:
        """判断是否为攻击流量"""
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        return src_ip in self.attacker_ips or dst_ip in self.victim_ips
    
    def _extract_packet_features(self, packet: Dict, prev_timestamp: float = 0) -> np.ndarray:
        """提取单个数据包的特征"""
        features = np.zeros(self.n_features, dtype=np.float32)
        
        # 数据包大小 (归一化到0-1)
        features[0] = min(packet.get('packet_size', 0) / 1500.0, 1.0)
        
        # 到达间隔时间 (归一化)
        timestamp = packet.get('timestamp', 0)
        iat = timestamp - prev_timestamp if prev_timestamp > 0 else 0
        features[1] = min(iat / 1.0, 1.0)  # 最大1秒
        
        # 协议
        protocol = packet.get('protocol', 6)
        features[2] = 1.0 if protocol == 6 else (0.5 if protocol == 17 else 0.0)
        
        # TCP标志
        features[3] = packet.get('tcp_flags', 0) / 255.0
        
        # 端口 (归一化)
        features[4] = packet.get('src_port', 0) / 65535.0
        features[5] = packet.get('dst_port', 0) / 65535.0
        
        # 方向
        features[6] = packet.get('direction', 0)
        
        # 负载和头部大小
        features[7] = min(packet.get('payload_size', 0) / 1500.0, 1.0)
        features[8] = min(packet.get('header_size', 20) / 60.0, 1.0)
        
        # 窗口大小
        features[9] = min(packet.get('window_size', 0) / 65535.0, 1.0)
        
        # TTL
        features[10] = packet.get('ttl', 64) / 255.0
        
        return features
    
    def process_packet(self, packet: Dict) -> Optional[Tuple[np.ndarray, int]]:
        """
        处理单个数据包
        
        Returns:
            如果流完成，返回 (特征矩阵, 标签)，否则返回None
        """
        flow_id = self._get_flow_id(packet)
        
        # 获取或创建流
        if flow_id not in self.flows:
            self.flows[flow_id] = FlowSample(
                flow_id=flow_id,
                label=1 if self._is_attack(packet) else 0
            )
        
        flow = self.flows[flow_id]
        flow.add_packet(packet)
        
        # 检查是否达到样本大小
        if flow.packet_count >= self.packets_per_flow:
            sample = self._create_sample(flow)
            del self.flows[flow_id]
            return sample, flow.label
        
        # 检查时间窗口
        if flow.duration >= self.time_window and flow.packet_count > 0:
            sample = self._create_sample(flow)
            del self.flows[flow_id]
            return sample, flow.label
        
        return None
    
    def _create_sample(self, flow: FlowSample) -> np.ndarray:
        """创建样本矩阵"""
        sample = np.zeros((self.packets_per_flow, self.n_features), dtype=np.float32)
        
        prev_timestamp = 0
        for i, packet in enumerate(flow.packets[:self.packets_per_flow]):
            sample[i] = self._extract_packet_features(packet, prev_timestamp)
            prev_timestamp = packet.get('timestamp', 0)
        
        return sample
    
    def flush_flows(self) -> List[Tuple[np.ndarray, int]]:
        """刷新所有未完成的流"""
        samples = []
        for flow in self.flows.values():
            if flow.packet_count > 0:
                sample = self._create_sample(flow)
                samples.append((sample, flow.label))
        self.flows.clear()
        return samples
    
    def parse_batch(self, packets: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量解析数据包
        
        Returns:
            (X, y) 特征矩阵和标签
        """
        samples = []
        labels = []
        
        for packet in packets:
            result = self.process_packet(packet)
            if result:
                samples.append(result[0])
                labels.append(result[1])
        
        # 刷新剩余流
        for sample, label in self.flush_flows():
            samples.append(sample)
            labels.append(label)
        
        if not samples:
            return np.array([]).reshape(0, self.packets_per_flow, self.n_features), np.array([])
        
        return np.array(samples), np.array(labels)
    
    def get_input_shape(self) -> Tuple[int, int]:
        """获取输入形状"""
        return (self.packets_per_flow, self.n_features)
