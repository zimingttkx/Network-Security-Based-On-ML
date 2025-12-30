"""
Slips流量分析器
分析网络流量特征
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlowFeatures:
    """流量特征"""
    # 基础信息
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 6
    
    # 时间特征
    duration: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    # 数据量特征
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    
    # 统计特征
    avg_packet_size: float = 0.0
    bytes_per_second: float = 0.0
    packets_per_second: float = 0.0
    
    # TCP特征
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.duration,
            self.bytes_sent,
            self.bytes_recv,
            self.packets_sent,
            self.packets_recv,
            self.avg_packet_size,
            self.bytes_per_second,
            self.packets_per_second,
            self.syn_count,
            self.ack_count,
            self.fin_count,
            self.rst_count,
            self.psh_count,
            self.protocol,
            self.src_port / 65535.0,
            self.dst_port / 65535.0
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict:
        return {
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'duration': self.duration,
            'bytes_sent': self.bytes_sent,
            'bytes_recv': self.bytes_recv,
            'packets_sent': self.packets_sent,
            'packets_recv': self.packets_recv
        }


class FlowAnalyzer:
    """
    流量分析器
    
    从原始数据包构建流量记录并提取特征
    """
    
    def __init__(self, flow_timeout: float = 60.0):
        """
        Args:
            flow_timeout: 流超时时间(秒)
        """
        self.flow_timeout = flow_timeout
        self.active_flows: Dict[str, FlowFeatures] = {}
        self.completed_flows: List[FlowFeatures] = []
    
    def _get_flow_key(self, packet: Dict) -> str:
        """生成流键"""
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        src_port = packet.get('src_port', 0)
        dst_port = packet.get('dst_port', 0)
        protocol = packet.get('protocol', 6)
        
        # 双向流
        if (src_ip, src_port) > (dst_ip, dst_port):
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
    
    def _is_forward(self, packet: Dict, flow: FlowFeatures) -> bool:
        """判断数据包方向"""
        return packet.get('src_ip') == flow.src_ip
    
    def process_packet(self, packet: Dict) -> Optional[FlowFeatures]:
        """
        处理数据包
        
        Returns:
            如果流完成，返回流特征
        """
        flow_key = self._get_flow_key(packet)
        timestamp = packet.get('timestamp', 0.0)
        
        # 检查超时流
        completed = self._check_timeouts(timestamp)
        
        # 获取或创建流
        if flow_key not in self.active_flows:
            flow = FlowFeatures(
                src_ip=packet.get('src_ip', ''),
                dst_ip=packet.get('dst_ip', ''),
                src_port=packet.get('src_port', 0),
                dst_port=packet.get('dst_port', 0),
                protocol=packet.get('protocol', 6),
                start_time=timestamp
            )
            self.active_flows[flow_key] = flow
        else:
            flow = self.active_flows[flow_key]
        
        # 更新流
        self._update_flow(flow, packet)
        
        # 检查是否完成 (FIN/RST)
        tcp_flags = packet.get('tcp_flags', 0)
        if tcp_flags & 0x01 or tcp_flags & 0x04:  # FIN or RST
            del self.active_flows[flow_key]
            self._finalize_flow(flow)
            return flow
        
        return completed[0] if completed else None
    
    def _update_flow(self, flow: FlowFeatures, packet: Dict):
        """更新流统计"""
        packet_size = packet.get('packet_size', 0)
        timestamp = packet.get('timestamp', 0.0)
        tcp_flags = packet.get('tcp_flags', 0)
        
        is_forward = self._is_forward(packet, flow)
        
        if is_forward:
            flow.bytes_sent += packet_size
            flow.packets_sent += 1
        else:
            flow.bytes_recv += packet_size
            flow.packets_recv += 1
        
        flow.end_time = timestamp
        flow.duration = flow.end_time - flow.start_time
        
        # TCP标志
        if tcp_flags & 0x02:
            flow.syn_count += 1
        if tcp_flags & 0x10:
            flow.ack_count += 1
        if tcp_flags & 0x01:
            flow.fin_count += 1
        if tcp_flags & 0x04:
            flow.rst_count += 1
        if tcp_flags & 0x08:
            flow.psh_count += 1
    
    def _finalize_flow(self, flow: FlowFeatures):
        """完成流计算"""
        total_packets = flow.packets_sent + flow.packets_recv
        total_bytes = flow.bytes_sent + flow.bytes_recv
        
        if total_packets > 0:
            flow.avg_packet_size = total_bytes / total_packets
        
        if flow.duration > 0:
            flow.bytes_per_second = total_bytes / flow.duration
            flow.packets_per_second = total_packets / flow.duration
        
        self.completed_flows.append(flow)
    
    def _check_timeouts(self, current_time: float) -> List[FlowFeatures]:
        """检查超时流"""
        completed = []
        expired_keys = []
        
        for key, flow in self.active_flows.items():
            if current_time - flow.end_time > self.flow_timeout:
                expired_keys.append(key)
                self._finalize_flow(flow)
                completed.append(flow)
        
        for key in expired_keys:
            del self.active_flows[key]
        
        return completed
    
    def flush(self) -> List[FlowFeatures]:
        """刷新所有活动流"""
        flows = []
        for flow in self.active_flows.values():
            self._finalize_flow(flow)
            flows.append(flow)
        self.active_flows.clear()
        return flows
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'active_flows': len(self.active_flows),
            'completed_flows': len(self.completed_flows)
        }
