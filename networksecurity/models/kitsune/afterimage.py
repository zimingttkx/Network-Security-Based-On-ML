"""
AfterImage - 增量统计特征提取器
基于 ymirsky/Kitsune-py 实现

AfterImage使用阻尼增量统计来跟踪网络流量的时间模式，
为每个数据包提取115维特征向量。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IncStat:
    """增量统计类 - 跟踪单个统计量的增量更新"""
    
    def __init__(self, lambda_: float = 1.0, init_time: float = 0.0, is_typed: bool = False):
        """
        Args:
            lambda_: 衰减因子，控制历史数据的权重衰减速度
            init_time: 初始时间戳
            is_typed: 是否按类型分组
        """
        self.lambda_ = lambda_
        self.is_typed = is_typed
        self.last_timestamp = init_time
        
        # 一阶统计量 (均值)
        self.weight = 0.0
        self.sum = 0.0
        
        # 二阶统计量 (方差)
        self.sum_sq = 0.0
        
        # 协方差相关
        self.src_sum = 0.0
        self.src_sum_sq = 0.0
        self.cov_sum = 0.0
    
    def insert(self, value: float, timestamp: float = 0.0, src_value: float = None):
        """插入新值并更新统计量"""
        # 计算时间衰减
        if timestamp > self.last_timestamp:
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
        
        # 更新统计量
        self.weight += 1.0
        self.sum += value
        self.sum_sq += value * value
        
        if src_value is not None:
            self.src_sum += src_value
            self.src_sum_sq += src_value * src_value
            self.cov_sum += value * src_value
    
    def mean(self) -> float:
        """计算加权均值"""
        if self.weight < 1e-10:
            return 0.0
        return self.sum / self.weight
    
    def var(self) -> float:
        """计算加权方差"""
        if self.weight < 2:
            return 0.0
        mean = self.mean()
        return max(0, self.sum_sq / self.weight - mean * mean)
    
    def std(self) -> float:
        """计算加权标准差"""
        return np.sqrt(self.var())
    
    def cov(self) -> float:
        """计算协方差"""
        if self.weight < 2:
            return 0.0
        return self.cov_sum / self.weight - (self.sum / self.weight) * (self.src_sum / self.weight)
    
    def pcc(self) -> float:
        """计算皮尔逊相关系数"""
        if self.weight < 2:
            return 0.0
        
        var_x = self.var()
        var_y = max(0, self.src_sum_sq / self.weight - (self.src_sum / self.weight) ** 2)
        
        if var_x < 1e-10 or var_y < 1e-10:
            return 0.0
        
        return self.cov() / (np.sqrt(var_x) * np.sqrt(var_y))
    
    def get_stats(self) -> Tuple[float, float, float]:
        """返回 (权重, 均值, 标准差)"""
        return self.weight, self.mean(), self.std()
    
    def get_stats_1d(self) -> List[float]:
        """返回一维统计特征"""
        return [self.weight, self.mean(), self.std()]
    
    def get_stats_2d(self) -> List[float]:
        """返回二维统计特征（包含协方差）"""
        return [self.weight, self.mean(), self.std(), self.cov(), self.pcc()]


class IncStatDB:
    """增量统计数据库 - 管理多个统计量"""
    
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.stats: Dict[str, IncStat] = {}
    
    def get_stat(self, key: str, init_time: float = 0.0) -> IncStat:
        """获取或创建统计量"""
        if key not in self.stats:
            self.stats[key] = IncStat(self.lambda_, init_time)
        return self.stats[key]
    
    def update(self, key: str, value: float, timestamp: float = 0.0):
        """更新统计量"""
        stat = self.get_stat(key, timestamp)
        stat.insert(value, timestamp)
    
    def get_stats(self, key: str) -> Tuple[float, float, float]:
        """获取统计量"""
        if key in self.stats:
            return self.stats[key].get_stats()
        return 0.0, 0.0, 0.0


class AfterImage:
    """
    AfterImage特征提取器
    
    为每个网络数据包提取115维特征向量，包括：
    - MAC层统计 (23特征)
    - IP层统计 (46特征)  
    - 传输层统计 (46特征)
    
    每组统计包含5个时间窗口的增量统计
    """
    
    # 时间窗口（秒）
    LAMBDAS = [5, 3, 1, 0.1, 0.01]
    
    def __init__(self, max_hosts: int = 100000):
        """
        Args:
            max_hosts: 最大跟踪主机数
        """
        self.max_hosts = max_hosts
        
        # 为每个时间窗口创建统计数据库
        self.mac_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        
        self.packet_count = 0
    
    def update_get_stats(self, src_mac: str, dst_mac: str, src_ip: str, dst_ip: str,
                         src_port: int, dst_port: int, packet_size: int, 
                         timestamp: float) -> np.ndarray:
        """
        更新统计并返回特征向量
        
        Args:
            src_mac: 源MAC地址
            dst_mac: 目标MAC地址
            src_ip: 源IP地址
            dst_ip: 目标IP地址
            src_port: 源端口
            dst_port: 目标端口
            packet_size: 数据包大小
            timestamp: 时间戳
            
        Returns:
            115维特征向量
        """
        self.packet_count += 1
        features = []
        
        # MAC层特征 (23维)
        mac_key = f"{src_mac}->{dst_mac}"
        mac_features = self._extract_channel_features(
            self.mac_stats, mac_key, packet_size, timestamp
        )
        features.extend(mac_features)
        
        # IP层特征 - 源IP (23维)
        ip_src_features = self._extract_channel_features(
            self.ip_stats, src_ip, packet_size, timestamp
        )
        features.extend(ip_src_features)
        
        # IP层特征 - IP对 (23维)
        ip_pair_key = f"{src_ip}->{dst_ip}"
        ip_pair_features = self._extract_channel_features(
            self.ip_pair_stats, ip_pair_key, packet_size, timestamp
        )
        features.extend(ip_pair_features)
        
        # Socket层特征 - 源Socket (23维)
        socket_src_key = f"{src_ip}:{src_port}"
        socket_src_features = self._extract_channel_features(
            self.socket_stats, socket_src_key, packet_size, timestamp
        )
        features.extend(socket_src_features)
        
        # Socket层特征 - Socket对 (23维)
        socket_pair_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        socket_pair_features = self._extract_channel_features(
            self.socket_pair_stats, socket_pair_key, packet_size, timestamp
        )
        features.extend(socket_pair_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_channel_features(self, stat_dbs: List[IncStatDB], key: str,
                                   value: float, timestamp: float) -> List[float]:
        """提取单个通道的特征"""
        features = []
        
        for db in stat_dbs:
            db.update(key, value, timestamp)
            weight, mean, std = db.get_stats(key)
            features.extend([weight, mean, std])
        
        # 添加额外的聚合特征
        all_weights = [db.get_stats(key)[0] for db in stat_dbs]
        features.extend([
            np.mean(all_weights),
            np.std(all_weights),
            np.max(all_weights) - np.min(all_weights) if all_weights else 0
        ])
        
        # 添加时间间隔特征
        features.extend([
            np.log1p(value),  # 对数包大小
            timestamp % 86400 / 86400,  # 日内时间归一化
            timestamp % 3600 / 3600,  # 小时内时间归一化
            timestamp % 60 / 60,  # 分钟内时间归一化
            1.0  # 占位符
        ])
        
        return features
    
    def extract_features_from_packet(self, packet_info: Dict) -> np.ndarray:
        """
        从数据包信息字典提取特征
        
        Args:
            packet_info: 包含以下键的字典:
                - src_mac, dst_mac
                - src_ip, dst_ip
                - src_port, dst_port
                - packet_size
                - timestamp
        """
        return self.update_get_stats(
            src_mac=packet_info.get('src_mac', '00:00:00:00:00:00'),
            dst_mac=packet_info.get('dst_mac', '00:00:00:00:00:00'),
            src_ip=packet_info.get('src_ip', '0.0.0.0'),
            dst_ip=packet_info.get('dst_ip', '0.0.0.0'),
            src_port=packet_info.get('src_port', 0),
            dst_port=packet_info.get('dst_port', 0),
            packet_size=packet_info.get('packet_size', 0),
            timestamp=packet_info.get('timestamp', 0.0)
        )
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return 115
    
    def reset(self):
        """重置所有统计"""
        self.mac_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.ip_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.socket_pair_stats = [IncStatDB(1.0 / l) for l in self.LAMBDAS]
        self.packet_count = 0
