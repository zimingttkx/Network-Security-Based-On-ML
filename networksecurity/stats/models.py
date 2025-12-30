"""
流量日志数据模型
定义流量日志的数据结构和枚举类型
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
import json


class ThreatType(str, Enum):
    """威胁类型枚举"""
    BENIGN = "benign"           # 正常流量
    PHISHING = "phishing"       # 钓鱼攻击
    MALWARE = "malware"         # 恶意软件
    DOS = "dos"                 # 拒绝服务攻击
    DDOS = "ddos"               # 分布式拒绝服务
    PROBE = "probe"             # 探测攻击
    R2L = "r2l"                 # 远程到本地攻击
    U2R = "u2r"                 # 用户到根攻击
    BRUTE_FORCE = "brute_force" # 暴力破解
    SQL_INJECTION = "sql_injection"  # SQL注入
    XSS = "xss"                 # 跨站脚本
    BOTNET = "botnet"           # 僵尸网络
    INFILTRATION = "infiltration"  # 渗透攻击
    WEB_ATTACK = "web_attack"   # Web攻击
    UNKNOWN = "unknown"         # 未知威胁


class ActionType(str, Enum):
    """处理动作枚举"""
    ALLOW = "allow"             # 放行
    BLOCK = "block"             # 阻断
    CHALLENGE = "challenge"     # 人机验证
    LOG = "log"                 # 仅记录
    ALERT = "alert"             # 告警


class RiskLevel(str, Enum):
    """风险等级枚举"""
    SAFE = "safe"               # 安全
    LOW = "low"                 # 低风险
    MEDIUM = "medium"           # 中风险
    HIGH = "high"               # 高风险
    CRITICAL = "critical"       # 严重


@dataclass
class GeoLocation:
    """地理位置信息"""
    country: str = "Unknown"
    country_code: str = "XX"
    city: str = "Unknown"
    latitude: float = 0.0
    longitude: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeoLocation':
        return cls(**data)


@dataclass
class ModelPrediction:
    """模型预测结果"""
    model_name: str
    model_type: str  # ml, dl, rl, pretrained
    score: float
    prediction: int
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPrediction':
        return cls(**data)


@dataclass
class TrafficLog:
    """流量日志数据模型"""
    # 基本信息
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 请求信息
    source_ip: str = ""
    source_port: int = 0
    dest_ip: str = ""
    dest_port: int = 0
    protocol: str = "HTTP"
    method: str = "GET"
    url: str = ""
    user_agent: str = ""
    
    # 检测结果
    threat_type: ThreatType = ThreatType.BENIGN
    risk_level: RiskLevel = RiskLevel.SAFE
    risk_score: float = 0.0
    action: ActionType = ActionType.ALLOW
    
    # 模型预测详情
    predictions: list = field(default_factory=list)
    ensemble_score: float = 0.0
    
    # 地理位置
    geo_location: Optional[GeoLocation] = None
    
    # 额外信息
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 人机验证
    captcha_required: bool = False
    captcha_passed: bool = False
    
    # 处理时间
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'source_ip': self.source_ip,
            'source_port': self.source_port,
            'dest_ip': self.dest_ip,
            'dest_port': self.dest_port,
            'protocol': self.protocol,
            'method': self.method,
            'url': self.url,
            'user_agent': self.user_agent,
            'threat_type': self.threat_type.value,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'action': self.action.value,
            'predictions': [p.to_dict() if isinstance(p, ModelPrediction) else p for p in self.predictions],
            'ensemble_score': self.ensemble_score,
            'geo_location': self.geo_location.to_dict() if self.geo_location else None,
            'features': self.features,
            'metadata': self.metadata,
            'captcha_required': self.captcha_required,
            'captcha_passed': self.captcha_passed,
            'processing_time_ms': self.processing_time_ms
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrafficLog':
        """从字典创建实例"""
        # 处理枚举类型
        if 'threat_type' in data and isinstance(data['threat_type'], str):
            data['threat_type'] = ThreatType(data['threat_type'])
        if 'risk_level' in data and isinstance(data['risk_level'], str):
            data['risk_level'] = RiskLevel(data['risk_level'])
        if 'action' in data and isinstance(data['action'], str):
            data['action'] = ActionType(data['action'])
        
        # 处理时间戳
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # 处理地理位置
        if 'geo_location' in data and data['geo_location'] and isinstance(data['geo_location'], dict):
            data['geo_location'] = GeoLocation.from_dict(data['geo_location'])
        
        # 处理预测结果
        if 'predictions' in data:
            data['predictions'] = [
                ModelPrediction.from_dict(p) if isinstance(p, dict) else p 
                for p in data['predictions']
            ]
        
        return cls(**data)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TrafficLog':
        """从JSON字符串创建实例"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class TrafficStats:
    """流量统计摘要"""
    total_requests: int = 0
    blocked_requests: int = 0
    allowed_requests: int = 0
    challenged_requests: int = 0
    
    threat_counts: Dict[str, int] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)
    risk_level_counts: Dict[str, int] = field(default_factory=dict)
    
    top_source_ips: list = field(default_factory=list)
    top_threat_types: list = field(default_factory=list)
    
    avg_risk_score: float = 0.0
    avg_processing_time_ms: float = 0.0
    
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'allowed_requests': self.allowed_requests,
            'challenged_requests': self.challenged_requests,
            'threat_counts': self.threat_counts,
            'action_counts': self.action_counts,
            'risk_level_counts': self.risk_level_counts,
            'top_source_ips': self.top_source_ips,
            'top_threat_types': self.top_threat_types,
            'avg_risk_score': self.avg_risk_score,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'time_range_start': self.time_range_start.isoformat() if self.time_range_start else None,
            'time_range_end': self.time_range_end.isoformat() if self.time_range_end else None
        }
