"""
统一数据预处理器
将各种输入格式转换为各算法所需的格式
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InputFormat(str, Enum):
    """输入数据格式"""
    RAW_PACKET = "raw_packet"      # 原始数据包
    FLOW = "flow"                   # 流量记录
    NSLKDD = "nslkdd"              # NSL-KDD格式
    CICIDS = "cicids"              # CICIDS格式
    FEATURE_VECTOR = "feature_vector"  # 特征向量


class OutputFormat(str, Enum):
    """输出数据格式"""
    KITSUNE = "kitsune"            # Kitsune 115维特征
    LUCID = "lucid"                # LUCID CNN输入
    SLIPS = "slips"                # Slips流量特征
    RL_STATE = "rl_state"          # RL状态向量
    UNIFIED = "unified"            # 统一特征格式


@dataclass
class ProcessedData:
    """处理后的数据"""
    features: np.ndarray
    format: OutputFormat
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedPreprocessor:
    """
    统一数据预处理器
    
    功能:
    1. 自动检测输入格式
    2. 转换为各算法所需格式
    3. 特征归一化和标准化
    4. 缺失值处理
    """
    
    # 标准特征维度
    FEATURE_DIMS = {
        OutputFormat.KITSUNE: 115,
        OutputFormat.LUCID: (10, 11),  # (time_steps, features)
        OutputFormat.SLIPS: 16,
        OutputFormat.RL_STATE: 15,
        OutputFormat.UNIFIED: 50
    }
    
    # NSL-KDD特征列
    NSLKDD_NUMERIC = [
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    
    def __init__(self):
        self.scaler_params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.is_fitted = False
    
    def detect_format(self, data: Any) -> InputFormat:
        """自动检测输入格式"""
        if isinstance(data, np.ndarray):
            return InputFormat.FEATURE_VECTOR
        
        if isinstance(data, dict):
            if 'duration' in data and 'src_bytes' in data:
                return InputFormat.NSLKDD
            if 'Flow Duration' in data:
                return InputFormat.CICIDS
            if 'src_ip' in data and 'dst_ip' in data:
                if 'packets_sent' in data or 'bytes_sent' in data:
                    return InputFormat.FLOW
                return InputFormat.RAW_PACKET
        
        return InputFormat.FEATURE_VECTOR
    
    def preprocess(self, data: Any, output_format: OutputFormat = OutputFormat.UNIFIED,
                   input_format: InputFormat = None) -> ProcessedData:
        """
        预处理数据
        
        Args:
            data: 输入数据
            output_format: 目标输出格式
            input_format: 输入格式(可选，自动检测)
        """
        if input_format is None:
            input_format = self.detect_format(data)
        
        # 转换为统一中间格式
        unified = self._to_unified(data, input_format)
        
        # 转换为目标格式
        if output_format == OutputFormat.UNIFIED:
            features = unified
        elif output_format == OutputFormat.KITSUNE:
            features = self._to_kitsune(unified)
        elif output_format == OutputFormat.LUCID:
            features = self._to_lucid(unified)
        elif output_format == OutputFormat.SLIPS:
            features = self._to_slips(unified)
        elif output_format == OutputFormat.RL_STATE:
            features = self._to_rl_state(unified)
        else:
            features = unified
        
        return ProcessedData(
            features=features,
            format=output_format,
            metadata={'input_format': input_format.value}
        )
    
    def _to_unified(self, data: Any, input_format: InputFormat) -> np.ndarray:
        """转换为统一中间格式 (50维)"""
        unified = np.zeros(50, dtype=np.float32)
        
        if input_format == InputFormat.FEATURE_VECTOR:
            arr = np.array(data, dtype=np.float32).flatten()
            unified[:min(len(arr), 50)] = arr[:50]
            
        elif input_format == InputFormat.RAW_PACKET:
            unified[0] = data.get('packet_size', 0) / 1500
            unified[1] = data.get('protocol', 6) / 255
            unified[2] = data.get('src_port', 0) / 65535
            unified[3] = data.get('dst_port', 0) / 65535
            unified[4] = data.get('tcp_flags', 0) / 255
            unified[5] = data.get('ttl', 64) / 255
            # IP地址编码
            src_ip = data.get('src_ip', '0.0.0.0')
            dst_ip = data.get('dst_ip', '0.0.0.0')
            unified[6:10] = [int(x)/255 for x in src_ip.split('.')]
            unified[10:14] = [int(x)/255 for x in dst_ip.split('.')]
            
        elif input_format == InputFormat.FLOW:
            unified[0] = min(data.get('duration', 0) / 60, 1.0)
            unified[1] = min(data.get('bytes_sent', 0) / 1e6, 1.0)
            unified[2] = min(data.get('bytes_recv', 0) / 1e6, 1.0)
            unified[3] = min(data.get('packets_sent', 0) / 1000, 1.0)
            unified[4] = min(data.get('packets_recv', 0) / 1000, 1.0)
            unified[5] = data.get('protocol', 6) / 255
            unified[6] = data.get('src_port', 0) / 65535
            unified[7] = data.get('dst_port', 0) / 65535
            
        elif input_format == InputFormat.NSLKDD:
            for i, feat in enumerate(self.NSLKDD_NUMERIC[:50]):
                if feat in data:
                    val = float(data[feat])
                    unified[i] = self._normalize_nslkdd_feature(feat, val)
                    
        elif input_format == InputFormat.CICIDS:
            unified[0] = min(data.get('Flow Duration', 0) / 1e6, 1.0)
            unified[1] = min(data.get('Total Fwd Packets', 0) / 1000, 1.0)
            unified[2] = min(data.get('Total Backward Packets', 0) / 1000, 1.0)
            unified[3] = min(data.get('Flow Bytes/s', 0) / 1e8, 1.0)
            unified[4] = min(data.get('Flow Packets/s', 0) / 1e4, 1.0)
        
        return unified
    
    def _normalize_nslkdd_feature(self, name: str, value: float) -> float:
        """归一化NSL-KDD特征"""
        ranges = {
            'duration': 60000, 'src_bytes': 1e9, 'dst_bytes': 1e9,
            'count': 500, 'srv_count': 500, 'dst_host_count': 255,
            'dst_host_srv_count': 255
        }
        max_val = ranges.get(name, 1.0)
        if max_val == 1.0 and '_rate' in name:
            return value  # 已经是0-1
        return min(value / max_val, 1.0)
    
    def _to_kitsune(self, unified: np.ndarray) -> np.ndarray:
        """转换为Kitsune格式 (115维)"""
        kitsune = np.zeros(115, dtype=np.float32)
        kitsune[:50] = unified
        # 填充额外特征
        kitsune[50:] = np.tile(unified[:13], 5)[:65]
        return kitsune
    
    def _to_lucid(self, unified: np.ndarray) -> np.ndarray:
        """转换为LUCID格式 (10, 11)"""
        lucid = np.zeros((10, 11), dtype=np.float32)
        # 将统一特征分布到时间步
        for t in range(10):
            lucid[t, :] = unified[t*5:(t*5)+11] if t*5+11 <= 50 else unified[:11]
        return lucid
    
    def _to_slips(self, unified: np.ndarray) -> np.ndarray:
        """转换为Slips格式 (16维)"""
        return unified[:16].copy()
    
    def _to_rl_state(self, unified: np.ndarray) -> np.ndarray:
        """转换为RL状态格式 (15维)"""
        return unified[:15].copy()
    
    def batch_preprocess(self, data_list: List[Any], 
                         output_format: OutputFormat = OutputFormat.UNIFIED) -> np.ndarray:
        """批量预处理"""
        results = [self.preprocess(d, output_format).features for d in data_list]
        return np.array(results)
    
    def fit(self, data: np.ndarray):
        """拟合归一化参数"""
        self.scaler_params['mean'] = np.mean(data, axis=0)
        self.scaler_params['std'] = np.std(data, axis=0)
        self.scaler_params['std'][self.scaler_params['std'] < 1e-10] = 1.0
        self.is_fitted = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用归一化"""
        if not self.is_fitted:
            return data
        return (data - self.scaler_params['mean']) / self.scaler_params['std']
