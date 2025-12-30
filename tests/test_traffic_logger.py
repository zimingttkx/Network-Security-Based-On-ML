"""
流量日志记录器单元测试
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta

from networksecurity.stats.models import (
    TrafficLog, ThreatType, ActionType, RiskLevel,
    GeoLocation, ModelPrediction
)
from networksecurity.stats.traffic_logger import TrafficLogger


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
def sample_log():
    """创建示例日志"""
    return TrafficLog(
        source_ip="192.168.1.100",
        source_port=54321,
        dest_ip="10.0.0.1",
        dest_port=80,
        protocol="HTTP",
        method="GET",
        url="http://example.com/login",
        user_agent="Mozilla/5.0",
        threat_type=ThreatType.PHISHING,
        risk_level=RiskLevel.HIGH,
        risk_score=0.85,
        action=ActionType.BLOCK,
        processing_time_ms=15.5
    )


class TestTrafficLoggerInit:
    """初始化测试"""
    
    def test_create_logger(self, temp_db):
        """测试创建日志记录器"""
        logger = TrafficLogger(db_path=temp_db)
        assert logger is not None
        assert logger.db_path == temp_db
        logger.close()
    
    def test_database_created(self, temp_db):
        """测试数据库文件创建"""
        logger = TrafficLogger(db_path=temp_db)
        assert os.path.exists(temp_db)
        logger.close()


class TestLogOperations:
    """日志操作测试"""
    
    def test_log_single(self, logger, sample_log):
        """测试记录单条日志"""
        log_id = logger.log(sample_log)
        assert log_id == sample_log.id
    
    def test_log_and_retrieve(self, logger, sample_log):
        """测试记录并检索日志"""
        logger.log(sample_log)
        retrieved = logger.get_by_id(sample_log.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_log.id
        assert retrieved.source_ip == sample_log.source_ip
        assert retrieved.threat_type == sample_log.threat_type
        assert retrieved.action == sample_log.action
        assert retrieved.risk_score == sample_log.risk_score
    
    def test_log_with_geo_location(self, logger):
        """测试带地理位置的日志"""
        log = TrafficLog(
            source_ip="8.8.8.8",
            geo_location=GeoLocation(
                country="USA",
                city="Mountain View",
                latitude=37.3861,
                longitude=-122.0839
            )
        )
        logger.log(log)
        retrieved = logger.get_by_id(log.id)
        
        assert retrieved.geo_location is not None
        assert retrieved.geo_location.country == "USA"
        assert retrieved.geo_location.city == "Mountain View"
    
    def test_log_with_predictions(self, logger):
        """测试带预测结果的日志"""
        predictions = [
            ModelPrediction(model_name="XGBoost", model_type="ml", score=0.8, prediction=1),
            ModelPrediction(model_name="DNN", model_type="dl", score=0.75, prediction=1)
        ]
        log = TrafficLog(
            source_ip="1.2.3.4",
            predictions=predictions,
            ensemble_score=0.78
        )
        logger.log(log)
        retrieved = logger.get_by_id(log.id)
        
        assert len(retrieved.predictions) == 2
        assert retrieved.predictions[0].model_name == "XGBoost"
        assert retrieved.ensemble_score == 0.78
    
    def test_log_batch(self, logger):
        """测试批量记录日志"""
        logs = [
            TrafficLog(source_ip=f"192.168.1.{i}", threat_type=ThreatType.BENIGN)
            for i in range(10)
        ]
        count = logger.log_batch(logs)
        assert count == 10
    
    def test_get_nonexistent(self, logger):
        """测试获取不存在的日志"""
        result = logger.get_by_id("nonexistent-id")
        assert result is None


class TestQueryOperations:
    """查询操作测试"""
    
    def test_query_all(self, logger):
        """测试查询所有日志"""
        for i in range(5):
            logger.log(TrafficLog(source_ip=f"10.0.0.{i}"))
        
        results = logger.query(limit=100)
        assert len(results) == 5
    
    def test_query_by_source_ip(self, logger):
        """测试按源IP查询"""
        logger.log(TrafficLog(source_ip="192.168.1.1"))
        logger.log(TrafficLog(source_ip="192.168.1.2"))
        logger.log(TrafficLog(source_ip="192.168.1.1"))
        
        results = logger.query(source_ip="192.168.1.1")
        assert len(results) == 2
    
    def test_query_by_threat_type(self, logger):
        """测试按威胁类型查询"""
        logger.log(TrafficLog(threat_type=ThreatType.BENIGN))
        logger.log(TrafficLog(threat_type=ThreatType.PHISHING))
        logger.log(TrafficLog(threat_type=ThreatType.PHISHING))
        logger.log(TrafficLog(threat_type=ThreatType.DOS))
        
        results = logger.query(threat_type=ThreatType.PHISHING)
        assert len(results) == 2
    
    def test_query_by_action(self, logger):
        """测试按动作查询"""
        logger.log(TrafficLog(action=ActionType.ALLOW))
        logger.log(TrafficLog(action=ActionType.BLOCK))
        logger.log(TrafficLog(action=ActionType.BLOCK))
        
        results = logger.query(action=ActionType.BLOCK)
        assert len(results) == 2
    
    def test_query_by_risk_level(self, logger):
        """测试按风险等级查询"""
        logger.log(TrafficLog(risk_level=RiskLevel.SAFE))
        logger.log(TrafficLog(risk_level=RiskLevel.HIGH))
        logger.log(TrafficLog(risk_level=RiskLevel.HIGH))
        
        results = logger.query(risk_level=RiskLevel.HIGH)
        assert len(results) == 2
    
    def test_query_by_min_risk_score(self, logger):
        """测试按最小风险分数查询"""
        logger.log(TrafficLog(risk_score=0.2))
        logger.log(TrafficLog(risk_score=0.5))
        logger.log(TrafficLog(risk_score=0.8))
        logger.log(TrafficLog(risk_score=0.9))
        
        results = logger.query(min_risk_score=0.7)
        assert len(results) == 2
    
    def test_query_by_time_range(self, logger):
        """测试按时间范围查询"""
        now = datetime.now()
        
        logger.log(TrafficLog(timestamp=now - timedelta(hours=2)))
        logger.log(TrafficLog(timestamp=now - timedelta(hours=1)))
        logger.log(TrafficLog(timestamp=now))
        
        results = logger.query(
            start_time=now - timedelta(hours=1, minutes=30),
            end_time=now + timedelta(minutes=1)
        )
        assert len(results) == 2
    
    def test_query_with_limit(self, logger):
        """测试限制返回数量"""
        for i in range(20):
            logger.log(TrafficLog(source_ip=f"10.0.0.{i}"))
        
        results = logger.query(limit=5)
        assert len(results) == 5
    
    def test_query_with_offset(self, logger):
        """测试分页偏移"""
        for i in range(10):
            logger.log(TrafficLog(source_ip=f"10.0.0.{i}"))
        
        results = logger.query(limit=5, offset=5)
        assert len(results) == 5
    
    def test_query_order_desc(self, logger):
        """测试降序排序"""
        logger.log(TrafficLog(source_ip="1.1.1.1", risk_score=0.1))
        logger.log(TrafficLog(source_ip="2.2.2.2", risk_score=0.9))
        logger.log(TrafficLog(source_ip="3.3.3.3", risk_score=0.5))
        
        results = logger.query(order_by="risk_score", order_desc=True)
        assert results[0].risk_score == 0.9
        assert results[-1].risk_score == 0.1


class TestCountOperations:
    """统计操作测试"""
    
    def test_count_all(self, logger):
        """测试统计所有日志"""
        for i in range(15):
            logger.log(TrafficLog())
        
        count = logger.count()
        assert count == 15
    
    def test_count_by_threat_type(self, logger):
        """测试按威胁类型统计"""
        logger.log(TrafficLog(threat_type=ThreatType.BENIGN))
        logger.log(TrafficLog(threat_type=ThreatType.PHISHING))
        logger.log(TrafficLog(threat_type=ThreatType.PHISHING))
        
        count = logger.count(threat_type=ThreatType.PHISHING)
        assert count == 2
    
    def test_count_by_action(self, logger):
        """测试按动作统计"""
        logger.log(TrafficLog(action=ActionType.ALLOW))
        logger.log(TrafficLog(action=ActionType.BLOCK))
        logger.log(TrafficLog(action=ActionType.BLOCK))
        logger.log(TrafficLog(action=ActionType.BLOCK))
        
        count = logger.count(action=ActionType.BLOCK)
        assert count == 3


class TestDeleteOperations:
    """删除操作测试"""
    
    def test_delete_old_logs(self, logger):
        """测试删除旧日志"""
        now = datetime.now()
        
        # 添加旧日志
        for i in range(5):
            logger.log(TrafficLog(timestamp=now - timedelta(days=40)))
        
        # 添加新日志
        for i in range(3):
            logger.log(TrafficLog(timestamp=now))
        
        deleted = logger.delete_old_logs(days=30)
        assert deleted == 5
        assert logger.count() == 3
    
    def test_clear_all(self, logger):
        """测试清空所有日志"""
        for i in range(10):
            logger.log(TrafficLog())
        
        assert logger.count() == 10
        deleted = logger.clear_all()
        assert deleted == 10
        assert logger.count() == 0


class TestDataIntegrity:
    """数据完整性测试"""
    
    def test_all_fields_preserved(self, logger):
        """测试所有字段保持完整"""
        original = TrafficLog(
            source_ip="192.168.1.100",
            source_port=54321,
            dest_ip="10.0.0.1",
            dest_port=443,
            protocol="HTTPS",
            method="POST",
            url="https://example.com/api",
            user_agent="CustomAgent/1.0",
            threat_type=ThreatType.SQL_INJECTION,
            risk_level=RiskLevel.CRITICAL,
            risk_score=0.95,
            action=ActionType.BLOCK,
            predictions=[
                ModelPrediction(model_name="RF", model_type="ml", score=0.9, prediction=1)
            ],
            ensemble_score=0.92,
            geo_location=GeoLocation(country="China", city="Beijing"),
            features={"feature1": 1, "feature2": 2},
            metadata={"key": "value"},
            captcha_required=True,
            captcha_passed=False,
            processing_time_ms=25.5
        )
        
        logger.log(original)
        retrieved = logger.get_by_id(original.id)
        
        assert retrieved.source_ip == original.source_ip
        assert retrieved.source_port == original.source_port
        assert retrieved.dest_ip == original.dest_ip
        assert retrieved.dest_port == original.dest_port
        assert retrieved.protocol == original.protocol
        assert retrieved.method == original.method
        assert retrieved.url == original.url
        assert retrieved.user_agent == original.user_agent
        assert retrieved.threat_type == original.threat_type
        assert retrieved.risk_level == original.risk_level
        assert retrieved.risk_score == original.risk_score
        assert retrieved.action == original.action
        assert retrieved.ensemble_score == original.ensemble_score
        assert retrieved.captcha_required == original.captcha_required
        assert retrieved.captcha_passed == original.captcha_passed
        assert retrieved.processing_time_ms == original.processing_time_ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
