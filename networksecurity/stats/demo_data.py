"""
模拟数据生成器
用于测试和演示
"""

import random
from datetime import datetime, timedelta

from networksecurity.stats.models import TrafficLog, ThreatType, ActionType, RiskLevel
from networksecurity.stats.traffic_logger import TrafficLogger


def generate_demo_data(logger: TrafficLogger, count: int = 500, hours_back: int = 24):
    """生成演示数据"""
    
    # 模拟IP池
    source_ips = [f"192.168.1.{i}" for i in range(1, 50)] + \
                 [f"10.0.0.{i}" for i in range(1, 30)] + \
                 ["45.33.32.156", "104.131.0.69", "185.199.108.153"]
    
    dest_ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "172.16.0.1", "172.16.0.2"]
    dest_ports = [80, 443, 8080, 22, 3306, 5432, 6379, 27017]
    
    threat_types = list(ThreatType)
    models = ["random_forest", "xgboost", "dnn", "lstm"]
    
    now = datetime.now()
    
    for i in range(count):
        # 随机时间（过去N小时内）
        random_seconds = random.randint(0, hours_back * 3600)
        timestamp = now - timedelta(seconds=random_seconds)
        
        # 生成请求数据
        is_threat = random.random() < 0.3  # 30%是威胁
        
        if is_threat:
            threat_type = random.choice([t for t in threat_types if t != ThreatType.BENIGN])
            risk_score = random.uniform(0.5, 1.0)
            action = random.choice([ActionType.BLOCK, ActionType.CHALLENGE, ActionType.ALERT])
            risk_level = random.choice([RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL])
        else:
            threat_type = ThreatType.BENIGN
            risk_score = random.uniform(0.0, 0.3)
            action = random.choice([ActionType.ALLOW, ActionType.LOG])
            risk_level = random.choice([RiskLevel.SAFE, RiskLevel.LOW])
        
        # 创建TrafficLog对象
        traffic_log = TrafficLog(
            timestamp=timestamp,
            source_ip=random.choice(source_ips),
            dest_ip=random.choice(dest_ips),
            dest_port=random.choice(dest_ports),
            threat_type=threat_type,
            action=action,
            risk_score=risk_score,
            risk_level=risk_level,
            processing_time_ms=random.uniform(1, 50)
        )
        
        logger.log(traffic_log)
    
    return count


if __name__ == "__main__":
    logger = TrafficLogger()
    count = generate_demo_data(logger, count=500)
    print(f"已生成 {count} 条演示数据")
