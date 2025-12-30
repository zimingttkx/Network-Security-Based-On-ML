"""
Slips检测器
整合行为分析、威胁情报和流量分析的完整检测器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time

from networksecurity.models.slips.behavior_analyzer import BehaviorAnalyzer, BehaviorProfile
from networksecurity.models.slips.threat_intelligence import ThreatIntelligence, ThreatCategory, IPReputation
from networksecurity.models.slips.flow_analyzer import FlowAnalyzer, FlowFeatures

logger = logging.getLogger(__name__)


@dataclass
class SlipsResult:
    """Slips检测结果"""
    is_threat: bool
    threat_score: float
    threat_types: List[str]
    src_ip: str
    dst_ip: str
    evidence: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'is_threat': self.is_threat,
            'threat_score': self.threat_score,
            'threat_types': self.threat_types,
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'evidence': self.evidence
        }


class SlipsDetector:
    """
    Slips风格入侵检测器
    
    整合多种检测方法:
    1. 行为分析: 检测异常行为模式
    2. 威胁情报: IP信誉查询
    3. 流量分析: 流量特征分析
    
    使用方法:
    ```python
    detector = SlipsDetector()
    for packet in packets:
        result = detector.process_packet(packet)
        if result.is_threat:
            print(f"威胁: {result.threat_types}")
    ```
    """
    
    def __init__(self, threat_threshold: float = 0.5):
        """
        Args:
            threat_threshold: 威胁判定阈值
        """
        self.threat_threshold = threat_threshold
        
        # 组件
        self.behavior_analyzer = BehaviorAnalyzer()
        self.threat_intel = ThreatIntelligence()
        self.flow_analyzer = FlowAnalyzer()
        
        # 统计
        self.total_packets = 0
        self.total_threats = 0
        self.threat_by_type: Dict[str, int] = {}
    
    def process_packet(self, packet: Dict) -> SlipsResult:
        """处理单个数据包"""
        self.total_packets += 1
        
        src_ip = packet.get('src_ip', '')
        dst_ip = packet.get('dst_ip', '')
        
        evidence = []
        threat_types = []
        threat_score = 0.0
        
        # 1. 威胁情报查询
        src_rep = self.threat_intel.query_ip(src_ip)
        dst_rep = self.threat_intel.query_ip(dst_ip)
        
        if src_rep.is_malicious():
            threat_score = max(threat_score, 1.0 - src_rep.score)
            threat_types.extend([c.value for c in src_rep.categories])
            evidence.append(f"源IP {src_ip} 在黑名单中")
        
        if dst_rep.is_malicious():
            threat_score = max(threat_score, 1.0 - dst_rep.score)
            threat_types.extend([c.value for c in dst_rep.categories])
            evidence.append(f"目标IP {dst_ip} 在黑名单中")
        
        # 2. 流量分析
        flow = self.flow_analyzer.process_packet(packet)
        
        # 3. 行为分析
        flow_dict = {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': packet.get('src_port', 0),
            'dst_port': packet.get('dst_port', 0),
            'protocol': packet.get('protocol', 6),
            'bytes_sent': packet.get('packet_size', 0),
            'bytes_recv': 0,
            'packets_sent': 1,
            'packets_recv': 0,
            'timestamp': packet.get('timestamp', time.time())
        }
        
        behavior_scores = self.behavior_analyzer.analyze_flow(flow_dict)
        
        for threat_type, score in behavior_scores.items():
            if score > 0.3:
                threat_score = max(threat_score, score)
                threat_types.append(threat_type)
                evidence.append(f"检测到{threat_type}行为 (分数={score:.2f})")
        
        # 判定是否为威胁
        is_threat = threat_score >= self.threat_threshold
        
        if is_threat:
            self.total_threats += 1
            for t in threat_types:
                self.threat_by_type[t] = self.threat_by_type.get(t, 0) + 1
            
            # 报告恶意IP
            if threat_score > 0.7:
                category = ThreatCategory.UNKNOWN
                if 'ddos' in threat_types:
                    category = ThreatCategory.BOTNET
                elif 'port_scan' in threat_types:
                    category = ThreatCategory.SCANNER
                elif 'c2' in threat_types:
                    category = ThreatCategory.C2
                self.threat_intel.report_ip(src_ip, category, threat_score)
        
        return SlipsResult(
            is_threat=is_threat,
            threat_score=threat_score,
            threat_types=list(set(threat_types)),
            src_ip=src_ip,
            dst_ip=dst_ip,
            evidence=evidence
        )
    
    def process_flow(self, flow: Dict) -> SlipsResult:
        """处理流量记录"""
        return self.process_packet(flow)
    
    def add_to_blacklist(self, ip: str, category: str = "unknown"):
        """添加IP到黑名单"""
        cat = ThreatCategory(category) if category in [c.value for c in ThreatCategory] else ThreatCategory.UNKNOWN
        self.threat_intel.add_to_blacklist(ip, cat)
    
    def add_to_whitelist(self, ip: str):
        """添加IP到白名单"""
        self.threat_intel.add_to_whitelist(ip)
    
    def get_suspicious_ips(self, threshold: float = 0.5) -> List[Dict]:
        """获取可疑IP列表"""
        return self.behavior_analyzer.get_suspicious_ips(threshold)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_packets': self.total_packets,
            'total_threats': self.total_threats,
            'threat_rate': self.total_threats / max(1, self.total_packets),
            'threat_by_type': self.threat_by_type,
            'intel_stats': self.threat_intel.get_stats(),
            'flow_stats': self.flow_analyzer.get_stats()
        }
    
    def reset(self):
        """重置检测器"""
        self.behavior_analyzer.reset()
        self.flow_analyzer = FlowAnalyzer()
        self.total_packets = 0
        self.total_threats = 0
        self.threat_by_type.clear()
