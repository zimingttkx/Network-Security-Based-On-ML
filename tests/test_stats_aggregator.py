"""
统计聚合器单元测试
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta

from networksecurity.stats.models import (
    TrafficLog, ThreatType, ActionType, RiskLevel, GeoLocation
)
from networksecurity.stats.traffic_logger import TrafficLogger
from networksecurity.stats.aggregator import StatsAggregator


@pytest.fixture
def temp_db():
    """创建临时数据库文件"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def logger(temp_db):
    """创建测试用的日志记录器"""
    logger = TrafficLogger(db_path=temp_db)
    yield logger
    logger.close()


@pytest.fixture
def aggregator(logger):
    """创建测试用的聚合器"""
    return StatsAggregator(logger)


@pytest.fixture
def populated_logger(logger):
    """填充测试数据的日志记录器"""
    now = datetime.now()
    
    # 添加各种类型的日志
    test_logs = [
        # 正常流量
        TrafficLog(source_ip="192.168.1.1", threat_type=ThreatType.BENIGN, 
                   action=ActionType.ALLOW, risk_score=0.1, timestamp=now),
        TrafficLog(source_ip="192.168.1.2", threat_type=ThreatType.BENIGN,
                   action=ActionType.ALLOW, risk_score=0.15, timestamp=now),
        TrafficLog(source_ip="192.168.1.1", threat_type=ThreatType.BENIGN,
                   action=ActionType.ALLOW, risk_score=0.2, timestamp=now),
        # 钓鱼攻击
        TrafficLog(source_ip="10.0.0.1", threat_type=ThreatType.PHISHING,
                   action=ActionType.BLOCK, risk_score=0.85, timestamp=now,
                   geo_location=GeoLocation(country="Russia")),
        TrafficLog(source_ip="10.0.0.2", threat_type=ThreatType.PHISHING,
                   action=ActionType.BLOCK, risk_score=0.9, timestamp=now,
                   geo_location=GeoLocation(country="China")),
        # DOS攻击
        TrafficLog(source_ip="10.0.0.1", threat_type=ThreatType.DOS,
                   action=ActionType.BLOCK, risk_score=0.95, timestamp=now,
                   geo_location=GeoLocation(country="Russia")),
        # 需要验证
        TrafficLog(source_ip="172.16.0.1", threat_type=ThreatType.UNKNOWN,
                   action=ActionType.CHALLENGE, risk_score=0.5, timestamp=now),
    ]
    
    for log in test_logs:
        logger.log(log)
    
    return logger


@pytest.fixture
def populated_aggregator(populated_logger):
    """使用填充数据的聚合器"""
    return StatsAggregator(populated_logger)


class TestGetOverview:
    """概览统计测试"""
    
    def test_empty_overview(self, aggregator):
        """测试空数据概览"""
        stats = aggregator.get_overview()
        assert stats.total_requests == 0
        assert stats.blocked_requests == 0
        assert stats.allowed_requests == 0
    
    def test_overview_counts(self, populated_aggregator):
        """测试概览计数"""
        stats = populated_aggregator.get_overview()
        assert stats.total_requests == 7
        assert stats.blocked_requests == 3
        assert stats.allowed_requests == 3
        assert stats.challenged_requests == 1
    
    def test_overview_threat_counts(self, populated_aggregator):
        """测试威胁类型计数"""
        stats = populated_aggregator.get_overview()
        assert stats.threat_counts['benign'] == 3
        assert stats.threat_counts['phishing'] == 2
        assert stats.threat_counts['dos'] == 1
    
    def test_overview_avg_risk_score(self, populated_aggregator):
        """测试平均风险分数"""
        stats = populated_aggregator.get_overview()
        # (0.1 + 0.15 + 0.2 + 0.85 + 0.9 + 0.95 + 0.5) / 7 ≈ 0.521
        assert 0.5 < stats.avg_risk_score < 0.55


class TestThreatDistribution:
    """威胁分布测试"""
    
    def test_empty_distribution(self, aggregator):
        """测试空数据分布"""
        dist = aggregator.get_threat_distribution()
        assert dist == {}
    
    def test_threat_distribution(self, populated_aggregator):
        """测试威胁分布"""
        dist = populated_aggregator.get_threat_distribution()
        assert dist['benign'] == 3
        assert dist['phishing'] == 2
        assert dist['dos'] == 1
        assert dist['unknown'] == 1


class TestActionDistribution:
    """动作分布测试"""
    
    def test_action_distribution(self, populated_aggregator):
        """测试动作分布"""
        dist = populated_aggregator.get_action_distribution()
        assert dist['allow'] == 3
        assert dist['block'] == 3
        assert dist['challenge'] == 1


class TestTopSourceIPs:
    """TOP IP测试"""
    
    def test_top_source_ips(self, populated_aggregator):
        """测试TOP源IP"""
        top_ips = populated_aggregator.get_top_source_ips(limit=5)
        # 192.168.1.1 出现2次, 10.0.0.1 出现2次
        assert len(top_ips) <= 5
        ip_dict = dict(top_ips)
        assert ip_dict.get('192.168.1.1', 0) == 2
        assert ip_dict.get('10.0.0.1', 0) == 2
    
    def test_top_source_ips_threat_only(self, populated_aggregator):
        """测试仅威胁流量的TOP IP"""
        top_ips = populated_aggregator.get_top_source_ips(threat_only=True)
        ip_dict = dict(top_ips)
        # 正常流量的IP不应该出现
        assert '192.168.1.1' not in ip_dict or ip_dict['192.168.1.1'] == 0


class TestTimeline:
    """时间线测试"""
    
    def test_timeline(self, logger):
        """测试时间线统计"""
        now = datetime.now()
        
        # 添加不同时间的日志
        logger.log(TrafficLog(timestamp=now - timedelta(hours=2)))
        logger.log(TrafficLog(timestamp=now - timedelta(hours=1)))
        logger.log(TrafficLog(timestamp=now - timedelta(hours=1), action=ActionType.BLOCK))
        logger.log(TrafficLog(timestamp=now))
        
        aggregator = StatsAggregator(logger)
        timeline = aggregator.get_timeline(
            start_time=now - timedelta(hours=3),
            interval_minutes=60
        )
        
        assert len(timeline) > 0
        # 检查每个时间桶都有统计数据
        for bucket in timeline:
            assert 'timestamp' in bucket
            assert 'total' in bucket
            assert 'blocked' in bucket


class TestGeoDistribution:
    """地理分布测试"""
    
    def test_geo_distribution(self, populated_aggregator):
        """测试地理分布"""
        dist = populated_aggregator.get_geo_distribution()
        assert dist.get('Russia', 0) == 2
        assert dist.get('China', 0) == 1
        assert dist.get('Unknown', 0) == 4  # 没有地理信息的


class TestRecentThreats:
    """最近威胁测试"""
    
    def test_recent_threats(self, populated_aggregator):
        """测试获取最近威胁"""
        threats = populated_aggregator.get_recent_threats(limit=10)
        # 应该只返回非正常流量
        for threat in threats:
            assert threat.threat_type != ThreatType.BENIGN
    
    def test_recent_threats_limit(self, populated_aggregator):
        """测试威胁数量限制"""
        threats = populated_aggregator.get_recent_threats(limit=2)
        assert len(threats) <= 2


class TestRiskScoreDistribution:
    """风险分数分布测试"""
    
    def test_risk_score_distribution(self, populated_aggregator):
        """测试风险分数分布"""
        dist = populated_aggregator.get_risk_score_distribution(bins=10)
        assert len(dist) == 10
        
        # 检查区间格式
        for bucket in dist:
            assert 'range' in bucket
            assert 'low' in bucket
            assert 'high' in bucket
            assert 'count' in bucket
            assert bucket['low'] < bucket['high']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
