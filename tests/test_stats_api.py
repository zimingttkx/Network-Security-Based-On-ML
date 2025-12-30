"""
统计API单元测试
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI

from networksecurity.stats.models import TrafficLog, ThreatType, ActionType, RiskLevel, GeoLocation
from networksecurity.stats.traffic_logger import TrafficLogger
from networksecurity.stats.aggregator import StatsAggregator
from networksecurity.stats import api as stats_api


@pytest.fixture
def temp_db():
    """创建临时数据库文件"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_app(temp_db):
    """创建测试应用"""
    # 重置全局实例
    stats_api._logger = TrafficLogger(db_path=temp_db)
    stats_api._aggregator = StatsAggregator(stats_api._logger)
    
    app = FastAPI()
    app.include_router(stats_api.router)
    
    yield app
    
    # 清理
    stats_api._logger.close()
    stats_api._logger = None
    stats_api._aggregator = None


@pytest.fixture
def client(test_app):
    """创建测试客户端"""
    return TestClient(test_app)


@pytest.fixture
def populated_app(test_app):
    """填充测试数据的应用"""
    logger = stats_api._logger
    now = datetime.now()
    
    # 添加测试数据
    test_logs = [
        TrafficLog(source_ip="192.168.1.1", threat_type=ThreatType.BENIGN,
                   action=ActionType.ALLOW, risk_score=0.1, timestamp=now),
        TrafficLog(source_ip="192.168.1.2", threat_type=ThreatType.BENIGN,
                   action=ActionType.ALLOW, risk_score=0.15, timestamp=now),
        TrafficLog(source_ip="10.0.0.1", threat_type=ThreatType.PHISHING,
                   action=ActionType.BLOCK, risk_score=0.85, timestamp=now,
                   geo_location=GeoLocation(country="Russia", city="Moscow")),
        TrafficLog(source_ip="10.0.0.2", threat_type=ThreatType.DOS,
                   action=ActionType.BLOCK, risk_score=0.95, timestamp=now,
                   geo_location=GeoLocation(country="China", city="Beijing")),
        TrafficLog(source_ip="172.16.0.1", threat_type=ThreatType.UNKNOWN,
                   action=ActionType.CHALLENGE, risk_score=0.5, timestamp=now),
    ]
    
    for log in test_logs:
        logger.log(log)
    
    return test_app


@pytest.fixture
def populated_client(populated_app):
    """使用填充数据的测试客户端"""
    return TestClient(populated_app)


class TestOverviewEndpoint:
    """概览端点测试"""
    
    def test_empty_overview(self, client):
        """测试空数据概览"""
        response = client.get("/api/v1/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 0
    
    def test_overview_with_data(self, populated_client):
        """测试有数据的概览"""
        response = populated_client.get("/api/v1/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 5
        assert data["blocked_requests"] == 2
        assert data["allowed_requests"] == 2
        assert data["challenged_requests"] == 1
    
    def test_overview_with_hours_param(self, populated_client):
        """测试带时间参数的概览"""
        response = populated_client.get("/api/v1/stats/overview?hours=1")
        assert response.status_code == 200


class TestThreatsEndpoint:
    """威胁分布端点测试"""
    
    def test_threat_distribution(self, populated_client):
        """测试威胁分布"""
        response = populated_client.get("/api/v1/stats/threats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "benign" in data["data"]
        assert data["data"]["benign"] == 2


class TestActionsEndpoint:
    """动作分布端点测试"""
    
    def test_action_distribution(self, populated_client):
        """测试动作分布"""
        response = populated_client.get("/api/v1/stats/actions")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["allow"] == 2
        assert data["data"]["block"] == 2


class TestSourcesEndpoint:
    """TOP源IP端点测试"""
    
    def test_top_sources(self, populated_client):
        """测试TOP源IP"""
        response = populated_client.get("/api/v1/stats/sources")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert len(data["data"]) > 0
    
    def test_top_sources_threat_only(self, populated_client):
        """测试仅威胁流量的TOP IP"""
        response = populated_client.get("/api/v1/stats/sources?threat_only=true")
        assert response.status_code == 200


class TestTimelineEndpoint:
    """时间线端点测试"""
    
    def test_timeline(self, populated_client):
        """测试时间线"""
        response = populated_client.get("/api/v1/stats/timeline")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True


class TestGeoEndpoint:
    """地理分布端点测试"""
    
    def test_geo_distribution(self, populated_client):
        """测试地理分布"""
        response = populated_client.get("/api/v1/stats/geo")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "Russia" in data["data"]


class TestRecentThreatsEndpoint:
    """最近威胁端点测试"""
    
    def test_recent_threats(self, populated_client):
        """测试最近威胁"""
        response = populated_client.get("/api/v1/stats/recent-threats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        # 应该只返回非正常流量
        for threat in data["data"]:
            assert threat["threat_type"] != "benign"


class TestLogsEndpoint:
    """日志查询端点测试"""
    
    def test_get_logs(self, populated_client):
        """测试获取日志"""
        response = populated_client.get("/api/v1/stats/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["total"] == 5
    
    def test_get_logs_with_filter(self, populated_client):
        """测试带过滤的日志查询"""
        response = populated_client.get("/api/v1/stats/logs?threat_type=phishing")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert len(data["data"]) == 1
    
    def test_create_log(self, client):
        """测试创建日志"""
        log_data = {
            "source_ip": "1.2.3.4",
            "dest_port": 80,
            "url": "http://test.com",
            "threat_type": "benign",
            "action": "allow",
            "risk_score": 0.1
        }
        response = client.post("/api/v1/stats/logs", json=log_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "id" in data
    
    def test_get_log_by_id(self, populated_client):
        """测试根据ID获取日志"""
        # 先获取日志列表
        response = populated_client.get("/api/v1/stats/logs?limit=1")
        logs = response.json()["data"]
        log_id = logs[0]["id"]
        
        # 根据ID获取
        response = populated_client.get(f"/api/v1/stats/logs/{log_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["id"] == log_id
    
    def test_get_nonexistent_log(self, client):
        """测试获取不存在的日志"""
        response = client.get("/api/v1/stats/logs/nonexistent-id")
        assert response.status_code == 404


class TestCleanupEndpoint:
    """清理端点测试"""
    
    def test_cleanup_old_logs(self, populated_client):
        """测试清理旧日志"""
        response = populated_client.delete("/api/v1/stats/logs/cleanup?days=30")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
