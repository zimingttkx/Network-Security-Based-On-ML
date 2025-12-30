"""
Kitsune特征提取器
从网络数据包或流量数据中提取特征
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PacketInfo:
    """数据包信息"""
    src_mac: str = "00:00:00:00:00:00"
    dst_mac: str = "00:00:00:00:00:00"
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 6  # TCP=6, UDP=17
    packet_size: int = 0
    timestamp: float = 0.0
    flags: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'src_mac': self.src_mac,
            'dst_mac': self.dst_mac,
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'packet_size': self.packet_size,
            'timestamp': self.timestamp,
            'flags': self.flags
        }


class KitsuneFeatureExtractor:
    """
    Kitsune特征提取器
    
    支持多种输入格式:
    1. 原始数据包信息 (PacketInfo)
    2. 流量特征向量 (已提取的特征)
    3. NSL-KDD/CICIDS格式的数据
    """
    
    # NSL-KDD特征列
    NSLKDD_FEATURES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    # CICIDS特征列 (部分)
    CICIDS_FEATURES = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count'
    ]
    
    def __init__(self, input_format: str = "auto"):
        """
        Args:
            input_format: 输入格式 ("auto", "packet", "nslkdd", "cicids", "raw")
        """
        self.input_format = input_format
        self.feature_dim = 115  # Kitsune默认特征维度
    
    def extract_from_packet(self, packet: PacketInfo) -> np.ndarray:
        """从数据包提取特征"""
        features = []
        
        # 基础特征
        features.append(packet.packet_size)
        features.append(packet.protocol)
        features.append(packet.src_port)
        features.append(packet.dst_port)
        features.append(packet.flags)
        
        # IP地址数值化
        src_ip_parts = [int(x) for x in packet.src_ip.split('.')]
        dst_ip_parts = [int(x) for x in packet.dst_ip.split('.')]
        features.extend(src_ip_parts)
        features.extend(dst_ip_parts)
        
        # 时间特征
        features.append(packet.timestamp % 86400)  # 日内秒数
        features.append(packet.timestamp % 3600)   # 小时内秒数
        
        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)
    
    def extract_from_nslkdd(self, row: Dict) -> np.ndarray:
        """从NSL-KDD格式数据提取特征"""
        features = []
        
        for feat in self.NSLKDD_FEATURES:
            if feat in row:
                val = row[feat]
                if isinstance(val, str):
                    val = hash(val) % 1000  # 简单编码
                features.append(float(val))
            else:
                features.append(0.0)
        
        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)
    
    def extract_from_cicids(self, row: Dict) -> np.ndarray:
        """从CICIDS格式数据提取特征"""
        features = []
        
        for feat in self.CICIDS_FEATURES:
            if feat in row:
                val = row[feat]
                if isinstance(val, str):
                    val = hash(val) % 1000
                features.append(float(val))
            else:
                features.append(0.0)
        
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)
    
    def extract(self, data: Any) -> np.ndarray:
        """自动检测格式并提取特征"""
        if isinstance(data, PacketInfo):
            return self.extract_from_packet(data)
        elif isinstance(data, np.ndarray):
            if len(data) >= self.feature_dim:
                return data[:self.feature_dim].astype(np.float32)
            else:
                padded = np.zeros(self.feature_dim, dtype=np.float32)
                padded[:len(data)] = data
                return padded
        elif isinstance(data, dict):
            if 'duration' in data or 'src_bytes' in data:
                return self.extract_from_nslkdd(data)
            elif 'Flow Duration' in data:
                return self.extract_from_cicids(data)
            else:
                return self.extract_from_packet(PacketInfo(**data))
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
            if len(arr) >= self.feature_dim:
                return arr[:self.feature_dim]
            else:
                padded = np.zeros(self.feature_dim, dtype=np.float32)
                padded[:len(arr)] = arr
                return padded
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def batch_extract(self, data_list: List[Any]) -> np.ndarray:
        """批量提取特征"""
        features = [self.extract(d) for d in data_list]
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        names = []
        for i in range(self.feature_dim):
            names.append(f"feature_{i}")
        return names
