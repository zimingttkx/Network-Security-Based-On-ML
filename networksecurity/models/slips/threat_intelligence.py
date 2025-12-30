"""
Slips威胁情报模块
管理IP信誉和威胁情报数据
"""

import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ThreatCategory(str, Enum):
    """威胁类别"""
    MALWARE = "malware"
    BOTNET = "botnet"
    SPAM = "spam"
    SCANNER = "scanner"
    BRUTEFORCE = "bruteforce"
    C2 = "c2"
    PHISHING = "phishing"
    TOR_EXIT = "tor_exit"
    PROXY = "proxy"
    UNKNOWN = "unknown"


@dataclass
class IPReputation:
    """IP信誉信息"""
    ip: str
    score: float = 0.5  # 0=恶意, 1=良好
    categories: Set[ThreatCategory] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    reports: int = 0
    source: str = "local"
    
    def is_malicious(self, threshold: float = 0.3) -> bool:
        return self.score < threshold
    
    def to_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'score': self.score,
            'categories': [c.value for c in self.categories],
            'is_malicious': self.is_malicious(),
            'reports': self.reports
        }


class ThreatIntelligence:
    """
    威胁情报管理器
    
    功能:
    - IP信誉查询
    - 黑名单/白名单管理
    - 威胁情报更新
    """
    
    # 已知恶意IP范围 (示例)
    KNOWN_MALICIOUS_RANGES = [
        '10.0.0.0/8',      # 私有网络 (测试用)
    ]
    
    # 已知良好IP
    KNOWN_GOOD_IPS = {
        '8.8.8.8',         # Google DNS
        '8.8.4.4',         # Google DNS
        '1.1.1.1',         # Cloudflare DNS
        '208.67.222.222',  # OpenDNS
    }
    
    def __init__(self):
        self.ip_cache: Dict[str, IPReputation] = {}
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set(self.KNOWN_GOOD_IPS)
        self.local_reports: Dict[str, List[Dict]] = {}
    
    def query_ip(self, ip: str) -> IPReputation:
        """查询IP信誉"""
        if ip in self.ip_cache:
            rep = self.ip_cache[ip]
            rep.last_seen = time.time()
            return rep
        
        # 创建新记录
        rep = IPReputation(ip=ip, first_seen=time.time(), last_seen=time.time())
        
        # 检查白名单
        if ip in self.whitelist:
            rep.score = 1.0
            rep.source = "whitelist"
        # 检查黑名单
        elif ip in self.blacklist:
            rep.score = 0.0
            rep.source = "blacklist"
        else:
            # 默认中性
            rep.score = 0.5
            rep.source = "unknown"
        
        self.ip_cache[ip] = rep
        return rep
    
    def report_ip(self, ip: str, category: ThreatCategory, confidence: float = 0.8):
        """报告恶意IP"""
        rep = self.query_ip(ip)
        rep.categories.add(category)
        rep.reports += 1
        
        # 更新分数 (降低)
        rep.score = max(0.0, rep.score - confidence * 0.2)
        
        # 保存报告
        if ip not in self.local_reports:
            self.local_reports[ip] = []
        self.local_reports[ip].append({
            'category': category.value,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def add_to_blacklist(self, ip: str, category: ThreatCategory = ThreatCategory.UNKNOWN):
        """添加到黑名单"""
        self.blacklist.add(ip)
        rep = self.query_ip(ip)
        rep.score = 0.0
        rep.categories.add(category)
        rep.source = "blacklist"
    
    def add_to_whitelist(self, ip: str):
        """添加到白名单"""
        self.whitelist.add(ip)
        if ip in self.blacklist:
            self.blacklist.remove(ip)
        rep = self.query_ip(ip)
        rep.score = 1.0
        rep.source = "whitelist"
    
    def get_malicious_ips(self, threshold: float = 0.3) -> List[IPReputation]:
        """获取恶意IP列表"""
        return [rep for rep in self.ip_cache.values() if rep.is_malicious(threshold)]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        malicious = len([r for r in self.ip_cache.values() if r.is_malicious()])
        return {
            'total_ips': len(self.ip_cache),
            'malicious_ips': malicious,
            'blacklist_size': len(self.blacklist),
            'whitelist_size': len(self.whitelist),
            'total_reports': sum(len(r) for r in self.local_reports.values())
        }
