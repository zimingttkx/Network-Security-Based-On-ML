"""
Slips行为分析器
基于 stratosphereips/StratosphereLinuxIPS 实现

分析网络实体的行为模式，检测异常行为。
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class BehaviorProfile:
    """行为配置文件"""
    ip: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    # 连接统计
    total_flows: int = 0
    total_bytes_sent: int = 0
    total_bytes_recv: int = 0
    total_packets_sent: int = 0
    total_packets_recv: int = 0
    
    # 端口统计
    unique_dst_ports: Set[int] = field(default_factory=set)
    unique_dst_ips: Set[str] = field(default_factory=set)
    unique_src_ports: Set[int] = field(default_factory=set)
    
    # 协议统计
    tcp_flows: int = 0
    udp_flows: int = 0
    icmp_flows: int = 0
    
    # 行为指标
    port_scan_score: float = 0.0
    ddos_score: float = 0.0
    c2_score: float = 0.0
    data_exfil_score: float = 0.0
    
    # 时间模式
    hourly_activity: List[int] = field(default_factory=lambda: [0] * 24)
    
    def update_flow(self, flow: Dict):
        """更新流量统计"""
        self.total_flows += 1
        self.total_bytes_sent += flow.get('bytes_sent', 0)
        self.total_bytes_recv += flow.get('bytes_recv', 0)
        self.total_packets_sent += flow.get('packets_sent', 0)
        self.total_packets_recv += flow.get('packets_recv', 0)
        
        dst_port = flow.get('dst_port', 0)
        dst_ip = flow.get('dst_ip', '')
        src_port = flow.get('src_port', 0)
        protocol = flow.get('protocol', 6)
        
        if dst_port:
            self.unique_dst_ports.add(dst_port)
        if dst_ip:
            self.unique_dst_ips.add(dst_ip)
        if src_port:
            self.unique_src_ports.add(src_port)
        
        if protocol == 6:
            self.tcp_flows += 1
        elif protocol == 17:
            self.udp_flows += 1
        elif protocol == 1:
            self.icmp_flows += 1
        
        timestamp = flow.get('timestamp', time.time())
        if self.first_seen == 0:
            self.first_seen = timestamp
        self.last_seen = timestamp
        
        hour = int((timestamp % 86400) / 3600)
        self.hourly_activity[hour] += 1
    
    def get_threat_score(self) -> float:
        """计算综合威胁分数"""
        return max(self.port_scan_score, self.ddos_score, 
                   self.c2_score, self.data_exfil_score)
    
    def to_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'total_flows': self.total_flows,
            'unique_dst_ports': len(self.unique_dst_ports),
            'unique_dst_ips': len(self.unique_dst_ips),
            'threat_score': self.get_threat_score(),
            'port_scan_score': self.port_scan_score,
            'ddos_score': self.ddos_score,
            'c2_score': self.c2_score
        }


class BehaviorAnalyzer:
    """
    行为分析器
    
    分析网络实体的行为模式，检测:
    - 端口扫描
    - DDoS攻击
    - C2通信
    - 数据外泄
    """
    
    # 检测阈值
    THRESHOLDS = {
        'port_scan_ports': 20,      # 扫描端口数阈值
        'port_scan_rate': 10,       # 每秒扫描端口数
        'ddos_pps': 1000,           # DDoS包速率
        'ddos_bps': 10_000_000,     # DDoS字节速率
        'c2_beacon_interval': 60,   # C2信标间隔(秒)
        'c2_beacon_variance': 0.1,  # C2信标方差
        'exfil_bytes': 1_000_000,   # 数据外泄字节阈值
    }
    
    def __init__(self):
        self.profiles: Dict[str, BehaviorProfile] = {}
        self.flow_history: List[Dict] = []
        self.max_history = 10000
    
    def get_profile(self, ip: str) -> BehaviorProfile:
        """获取或创建IP配置文件"""
        if ip not in self.profiles:
            self.profiles[ip] = BehaviorProfile(ip=ip)
        return self.profiles[ip]
    
    def analyze_flow(self, flow: Dict) -> Dict[str, float]:
        """
        分析单个流量
        
        Returns:
            各类威胁的分数
        """
        src_ip = flow.get('src_ip', '')
        profile = self.get_profile(src_ip)
        profile.update_flow(flow)
        
        # 保存历史
        self.flow_history.append(flow)
        if len(self.flow_history) > self.max_history:
            self.flow_history = self.flow_history[-self.max_history:]
        
        # 分析各类威胁
        scores = {
            'port_scan': self._detect_port_scan(profile),
            'ddos': self._detect_ddos(profile, flow),
            'c2': self._detect_c2(profile),
            'data_exfil': self._detect_data_exfil(profile)
        }
        
        # 更新配置文件分数
        profile.port_scan_score = scores['port_scan']
        profile.ddos_score = scores['ddos']
        profile.c2_score = scores['c2']
        profile.data_exfil_score = scores['data_exfil']
        
        return scores
    
    def _detect_port_scan(self, profile: BehaviorProfile) -> float:
        """检测端口扫描"""
        n_ports = len(profile.unique_dst_ports)
        n_ips = len(profile.unique_dst_ips)
        
        if profile.total_flows < 5:
            return 0.0
        
        # 端口扫描特征: 大量不同端口，少量目标IP
        if n_ports > self.THRESHOLDS['port_scan_ports']:
            port_ratio = n_ports / max(1, profile.total_flows)
            ip_ratio = n_ips / max(1, profile.total_flows)
            
            if port_ratio > 0.5 and ip_ratio < 0.3:
                return min(1.0, n_ports / 100)
        
        return 0.0
    
    def _detect_ddos(self, profile: BehaviorProfile, flow: Dict) -> float:
        """检测DDoS攻击"""
        duration = profile.last_seen - profile.first_seen
        if duration < 1:
            return 0.0
        
        pps = profile.total_packets_sent / duration
        bps = profile.total_bytes_sent / duration
        
        score = 0.0
        if pps > self.THRESHOLDS['ddos_pps']:
            score = max(score, min(1.0, pps / (self.THRESHOLDS['ddos_pps'] * 10)))
        if bps > self.THRESHOLDS['ddos_bps']:
            score = max(score, min(1.0, bps / (self.THRESHOLDS['ddos_bps'] * 10)))
        
        # 检查是否针对单一目标
        if len(profile.unique_dst_ips) == 1 and profile.total_flows > 100:
            score *= 1.5
        
        return min(1.0, score)
    
    def _detect_c2(self, profile: BehaviorProfile) -> float:
        """检测C2通信"""
        if profile.total_flows < 10:
            return 0.0
        
        # C2特征: 规律的通信间隔
        # 简化实现: 检查是否有规律的活动模式
        active_hours = sum(1 for h in profile.hourly_activity if h > 0)
        
        if active_hours >= 20:  # 几乎全天活动
            return 0.3
        
        return 0.0
    
    def _detect_data_exfil(self, profile: BehaviorProfile) -> float:
        """检测数据外泄"""
        if profile.total_bytes_sent > self.THRESHOLDS['exfil_bytes']:
            # 发送远大于接收
            ratio = profile.total_bytes_sent / max(1, profile.total_bytes_recv)
            if ratio > 10:
                return min(1.0, ratio / 100)
        return 0.0
    
    def get_suspicious_ips(self, threshold: float = 0.5) -> List[Dict]:
        """获取可疑IP列表"""
        suspicious = []
        for ip, profile in self.profiles.items():
            score = profile.get_threat_score()
            if score >= threshold:
                suspicious.append(profile.to_dict())
        return sorted(suspicious, key=lambda x: x['threat_score'], reverse=True)
    
    def reset(self):
        """重置分析器"""
        self.profiles.clear()
        self.flow_history.clear()
