"""
流量日志数据模型单元测试
"""

import pytest
from datetime import datetime
import json

from networksecurity.stats.models import (
    ThreatType, ActionType, RiskLevel,
    GeoLocation, ModelPrediction, TrafficLog, TrafficStats
)


class TestThreatType:
    """威胁类型枚举测试"""
    
    def test_threat_type_values(self):
        """测试威胁类型枚举值"""
        assert ThreatType.BENIGN.value == "benign"
        assert ThreatType.PHISHING.value == "phishing"
        assert ThreatType.DOS.value == "dos"
        assert ThreatType.DDOS.value == "ddos"
        assert ThreatType.UNKNOWN.value == "unknown"
    
    def test_threat_type_from_string(self):
        """测试从字符串创建枚举"""
        assert ThreatType("benign") == ThreatType.BENIGN
        assert ThreatType("phishing") == ThreatType.PHISHING
    
    def test_threat_type_invalid(self):
        """测试无效值"""
        with pytest.raises(ValueError):
            ThreatType("invalid_type")


class TestActionType:
    """处理动作枚举测试"""
    
    def test_action_type_values(self):
        """测试动作类型枚举值"""
        assert ActionType.ALLOW.value == "allow"
        assert ActionType.BLOCK.value == "block"
        assert ActionType.CHALLENGE.value == "challenge"
        assert ActionType.LOG.value == "log"
        assert ActionType.ALERT.value == "alert"


class TestRiskLevel:
    """风险等级枚举测试"""
    
    def test_risk_level_values(self):
        """测试风险等级枚举值"""
        assert RiskLevel.SAFE.value == "safe"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestGeoLocation:
    """地理位置测试"""
    
    def test_default_values(self):
        """测试默认值"""
        geo = GeoLocation()
        assert geo.country == "Unknown"
        assert geo.country_code == "XX"
        assert geo.city == "Unknown"
        assert geo.latitude == 0.0
        assert geo.longitude == 0.0
    
    def test_custom_values(self):
        """测试自定义值"""
        geo = GeoLocation(
            country="China",
            country_code="CN",
            city="Beijing",
            latitude=39.9042,
            longitude=116.4074
        )
        assert geo.country == "China"
        assert geo.country_code == "CN"
        assert geo.city == "Beijing"
    
    def test_to_dict(self):
        """测试转换为字典"""
        geo = GeoLocation(country="USA", country_code="US", city="New York")
        data = geo.to_dict()
        assert data['country'] == "USA"
        assert data['country_code'] == "US"
        assert data['city'] == "New York"
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'country': 'Japan',
            'country_code': 'JP',
            'city': 'Tokyo',
            'latitude': 35.6762,
            'longitude': 139.6503
        }
        geo = GeoLocation.from_dict(data)
        assert geo.country == "Japan"
        assert geo.city == "Tokyo"


class TestModelPrediction:
    """模型预测结果测试"""
    
    def test_creation(self):
        """测试创建预测结果"""
        pred = ModelPrediction(
            model_name="XGBoost",
            model_type="ml",
            score=0.85,
            prediction=1,
            confidence=0.92,
            inference_time_ms=5.2
        )
        assert pred.model_name == "XGBoost"
        assert pred.model_type == "ml"
        assert pred.score == 0.85
        assert pred.prediction == 1
        assert pred.confidence == 0.92
    
    def test_to_dict(self):
        """测试转换为字典"""
        pred = ModelPrediction(
            model_name="DNN",
            model_type="dl",
            score=0.78,
            prediction=0
        )
        data = pred.to_dict()
        assert data['model_name'] == "DNN"
        assert data['model_type'] == "dl"
        assert data['score'] == 0.78
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'model_name': 'LSTM',
            'model_type': 'dl',
            'score': 0.91,
            'prediction': 1,
            'confidence': 0.88,
            'inference_time_ms': 12.5
        }
        pred = ModelPrediction.from_dict(data)
        assert pred.model_name == "LSTM"
        assert pred.score == 0.91


class TestTrafficLog:
    """流量日志测试"""
    
    def test_default_creation(self):
        """测试默认创建"""
        log = TrafficLog()
        assert log.id is not None
        assert log.timestamp is not None
        assert log.threat_type == ThreatType.BENIGN
        assert log.action == ActionType.ALLOW
        assert log.risk_level == RiskLevel.SAFE
    
    def test_custom_creation(self):
        """测试自定义创建"""
        log = TrafficLog(
            source_ip="192.168.1.100",
            source_port=54321,
            dest_ip="10.0.0.1",
            dest_port=80,
            url="http://example.com/login",
            threat_type=ThreatType.PHISHING,
            risk_level=RiskLevel.HIGH,
            risk_score=0.85,
            action=ActionType.BLOCK
        )
        assert log.source_ip == "192.168.1.100"
        assert log.dest_port == 80
        assert log.threat_type == ThreatType.PHISHING
        assert log.action == ActionType.BLOCK
    
    def test_with_predictions(self):
        """测试带预测结果的日志"""
        predictions = [
            ModelPrediction(model_name="XGBoost", model_type="ml", score=0.8, prediction=1),
            ModelPrediction(model_name="DNN", model_type="dl", score=0.75, prediction=1)
        ]
        log = TrafficLog(
            source_ip="10.0.0.50",
            predictions=predictions,
            ensemble_score=0.78
        )
        assert len(log.predictions) == 2
        assert log.ensemble_score == 0.78
    
    def test_with_geo_location(self):
        """测试带地理位置的日志"""
        geo = GeoLocation(country="China", city="Shanghai")
        log = TrafficLog(
            source_ip="1.2.3.4",
            geo_location=geo
        )
        assert log.geo_location.country == "China"
        assert log.geo_location.city == "Shanghai"
    
    def test_to_dict(self):
        """测试转换为字典"""
        log = TrafficLog(
            source_ip="192.168.1.1",
            threat_type=ThreatType.DOS,
            action=ActionType.BLOCK,
            risk_score=0.95
        )
        data = log.to_dict()
        assert data['source_ip'] == "192.168.1.1"
        assert data['threat_type'] == "dos"
        assert data['action'] == "block"
        assert data['risk_score'] == 0.95
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'id': 'test-id-123',
            'timestamp': '2024-01-15T10:30:00',
            'source_ip': '8.8.8.8',
            'source_port': 443,
            'threat_type': 'malware',
            'risk_level': 'critical',
            'action': 'block',
            'risk_score': 0.99,
            'predictions': [],
            'features': {},
            'metadata': {}
        }
        log = TrafficLog.from_dict(data)
        assert log.id == 'test-id-123'
        assert log.source_ip == '8.8.8.8'
        assert log.threat_type == ThreatType.MALWARE
        assert log.risk_level == RiskLevel.CRITICAL
        assert log.action == ActionType.BLOCK
    
    def test_to_json(self):
        """测试转换为JSON"""
        log = TrafficLog(
            source_ip="1.1.1.1",
            threat_type=ThreatType.XSS,
            action=ActionType.ALERT
        )
        json_str = log.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['source_ip'] == "1.1.1.1"
        assert data['threat_type'] == "xss"
    
    def test_from_json(self):
        """测试从JSON创建"""
        json_str = '''
        {
            "id": "json-test-id",
            "timestamp": "2024-06-01T12:00:00",
            "source_ip": "5.5.5.5",
            "source_port": 8080,
            "dest_ip": "",
            "dest_port": 0,
            "protocol": "HTTP",
            "method": "POST",
            "url": "/api/login",
            "user_agent": "Mozilla/5.0",
            "threat_type": "brute_force",
            "risk_level": "high",
            "risk_score": 0.88,
            "action": "challenge",
            "predictions": [],
            "ensemble_score": 0.0,
            "geo_location": null,
            "features": {},
            "metadata": {},
            "captcha_required": true,
            "captcha_passed": false,
            "processing_time_ms": 15.5
        }
        '''
        log = TrafficLog.from_json(json_str)
        assert log.id == "json-test-id"
        assert log.source_ip == "5.5.5.5"
        assert log.threat_type == ThreatType.BRUTE_FORCE
        assert log.action == ActionType.CHALLENGE
        assert log.captcha_required == True
    
    def test_round_trip_serialization(self):
        """测试序列化往返"""
        original = TrafficLog(
            source_ip="10.20.30.40",
            dest_ip="192.168.0.1",
            dest_port=443,
            url="https://suspicious-site.com",
            threat_type=ThreatType.PHISHING,
            risk_level=RiskLevel.HIGH,
            risk_score=0.92,
            action=ActionType.BLOCK,
            predictions=[
                ModelPrediction(model_name="RF", model_type="ml", score=0.9, prediction=1)
            ],
            geo_location=GeoLocation(country="Russia", city="Moscow"),
            processing_time_ms=25.3
        )
        
        # 转换为JSON再转回来
        json_str = original.to_json()
        restored = TrafficLog.from_json(json_str)
        
        assert restored.source_ip == original.source_ip
        assert restored.threat_type == original.threat_type
        assert restored.risk_score == original.risk_score
        assert restored.geo_location.country == "Russia"
        assert len(restored.predictions) == 1


class TestTrafficStats:
    """流量统计摘要测试"""
    
    def test_default_creation(self):
        """测试默认创建"""
        stats = TrafficStats()
        assert stats.total_requests == 0
        assert stats.blocked_requests == 0
        assert stats.allowed_requests == 0
        assert stats.threat_counts == {}
    
    def test_custom_creation(self):
        """测试自定义创建"""
        stats = TrafficStats(
            total_requests=1000,
            blocked_requests=50,
            allowed_requests=900,
            challenged_requests=50,
            threat_counts={'phishing': 30, 'dos': 20},
            avg_risk_score=0.15
        )
        assert stats.total_requests == 1000
        assert stats.blocked_requests == 50
        assert stats.threat_counts['phishing'] == 30
    
    def test_to_dict(self):
        """测试转换为字典"""
        stats = TrafficStats(
            total_requests=500,
            blocked_requests=25,
            top_source_ips=[('1.1.1.1', 100), ('2.2.2.2', 50)]
        )
        data = stats.to_dict()
        assert data['total_requests'] == 500
        assert data['blocked_requests'] == 25
        assert len(data['top_source_ips']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
