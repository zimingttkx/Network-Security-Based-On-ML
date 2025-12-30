"""
防火墙模块单元测试
"""

import pytest
import numpy as np
import time

from networksecurity.firewall.detector import (
    ThreatDetector, DetectionResult, ThreatLevel, ActionType
)
from networksecurity.firewall.captcha import (
    CaptchaService, CaptchaChallenge, CaptchaType
)


class MockModel:
    """模拟模型"""
    def __init__(self, threat_prob: float = 0.5):
        self.threat_prob = threat_prob
    
    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else 1
        return np.array([[1 - self.threat_prob, self.threat_prob]] * n)
    
    def predict(self, X):
        return np.array([1 if self.threat_prob > 0.5 else 0])


class TestThreatLevel:
    """威胁等级测试"""
    
    def test_threat_levels(self):
        assert ThreatLevel.SAFE == "safe"
        assert ThreatLevel.CRITICAL == "critical"


class TestActionType:
    """动作类型测试"""
    
    def test_action_types(self):
        assert ActionType.ALLOW == "allow"
        assert ActionType.BLOCK == "block"


class TestDetectionResult:
    """检测结果测试"""
    
    def test_result_creation(self):
        result = DetectionResult(
            is_threat=True,
            threat_level=ThreatLevel.HIGH,
            confidence=0.85,
            action=ActionType.CHALLENGE,
            model_name="test"
        )
        assert result.is_threat
        assert result.confidence == 0.85
    
    def test_result_to_dict(self):
        result = DetectionResult(
            is_threat=False,
            threat_level=ThreatLevel.SAFE,
            confidence=0.1,
            action=ActionType.ALLOW,
            model_name="test"
        )
        d = result.to_dict()
        assert d['is_threat'] == False
        assert d['threat_level'] == 'safe'


class TestThreatDetector:
    """威胁检测器测试"""
    
    def test_detector_creation(self):
        detector = ThreatDetector()
        assert detector.default_model is None
        assert len(detector.models) == 0
    
    def test_register_model(self):
        detector = ThreatDetector()
        model = MockModel()
        detector.register_model("test_model", model)
        assert "test_model" in detector.models
        assert detector.default_model == "test_model"
    
    def test_unregister_model(self):
        detector = ThreatDetector()
        detector.register_model("test", MockModel())
        detector.unregister_model("test")
        assert "test" not in detector.models
    
    def test_detect_safe(self):
        detector = ThreatDetector()
        detector.register_model("safe_model", MockModel(threat_prob=0.1))
        features = {'f1': 0.5, 'f2': 0.3}
        result = detector.detect(features)
        assert result.threat_level == ThreatLevel.SAFE
        assert result.action == ActionType.ALLOW
    
    def test_detect_threat(self):
        detector = ThreatDetector()
        detector.register_model("threat_model", MockModel(threat_prob=0.9))
        features = np.array([0.5, 0.3, 0.8])
        result = detector.detect(features)
        assert result.is_threat
        assert result.threat_level == ThreatLevel.HIGH
    
    def test_detect_critical(self):
        detector = ThreatDetector()
        detector.register_model("critical_model", MockModel(threat_prob=0.98))
        result = detector.detect({'f1': 1.0})
        assert result.threat_level == ThreatLevel.CRITICAL
        assert result.action == ActionType.BLOCK
    
    def test_detect_batch(self):
        detector = ThreatDetector()
        detector.register_model("model", MockModel(threat_prob=0.5))
        features_list = [{'f1': 0.1}, {'f2': 0.2}, {'f3': 0.3}]
        results = detector.detect_batch(features_list)
        assert len(results) == 3
    
    def test_fallback_detection(self):
        detector = ThreatDetector()
        result = detector.detect({'f1': 0.5})
        assert result.model_name == "fallback"
        assert result.action == ActionType.ALLOW
    
    def test_list_models(self):
        detector = ThreatDetector()
        detector.register_model("m1", MockModel())
        detector.register_model("m2", MockModel())
        models = detector.list_models()
        assert "m1" in models
        assert "m2" in models
    
    def test_get_model_info(self):
        detector = ThreatDetector()
        detector.register_model("test", MockModel())
        info = detector.get_model_info()
        assert 'models' in info
        assert 'default_model' in info


class TestCaptchaType:
    """验证码类型测试"""
    
    def test_captcha_types(self):
        assert CaptchaType.MATH == "math"
        assert CaptchaType.TEXT == "text"


class TestCaptchaChallenge:
    """验证码挑战测试"""
    
    def test_challenge_creation(self):
        import hashlib
        answer_hash = hashlib.sha256("42".encode()).hexdigest()
        challenge = CaptchaChallenge(
            challenge_id="test123",
            challenge_type=CaptchaType.MATH,
            question="6 * 7 = ?",
            answer_hash=answer_hash,
            created_at=time.time(),
            expires_at=time.time() + 300
        )
        assert challenge.challenge_id == "test123"
    
    def test_challenge_verify_correct(self):
        import hashlib
        answer_hash = hashlib.sha256("42".encode()).hexdigest()
        challenge = CaptchaChallenge(
            challenge_id="test",
            challenge_type=CaptchaType.MATH,
            question="6 * 7 = ?",
            answer_hash=answer_hash,
            created_at=time.time(),
            expires_at=time.time() + 300
        )
        assert challenge.verify("42")
    
    def test_challenge_verify_wrong(self):
        import hashlib
        answer_hash = hashlib.sha256("42".encode()).hexdigest()
        challenge = CaptchaChallenge(
            challenge_id="test",
            challenge_type=CaptchaType.MATH,
            question="6 * 7 = ?",
            answer_hash=answer_hash,
            created_at=time.time(),
            expires_at=time.time() + 300
        )
        assert not challenge.verify("41")
    
    def test_challenge_expired(self):
        import hashlib
        answer_hash = hashlib.sha256("42".encode()).hexdigest()
        challenge = CaptchaChallenge(
            challenge_id="test",
            challenge_type=CaptchaType.MATH,
            question="test",
            answer_hash=answer_hash,
            created_at=time.time() - 400,
            expires_at=time.time() - 100
        )
        assert challenge.is_expired()
        assert not challenge.verify("42")


class TestCaptchaService:
    """验证码服务测试"""
    
    def test_service_creation(self):
        service = CaptchaService(expiry_seconds=60)
        assert service.expiry_seconds == 60
    
    def test_generate_math_challenge(self):
        service = CaptchaService()
        challenge = service.generate_challenge(CaptchaType.MATH)
        assert challenge.challenge_type == CaptchaType.MATH
        assert "=" in challenge.question
    
    def test_generate_text_challenge(self):
        service = CaptchaService()
        challenge = service.generate_challenge(CaptchaType.TEXT)
        assert challenge.challenge_type == CaptchaType.TEXT
    
    def test_verify_success(self):
        service = CaptchaService()
        # 手动创建可预测的验证码
        import hashlib
        answer = "42"
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        challenge = CaptchaChallenge(
            challenge_id="test_verify",
            challenge_type=CaptchaType.MATH,
            question="6 * 7 = ?",
            answer_hash=answer_hash,
            created_at=time.time(),
            expires_at=time.time() + 300
        )
        service.challenges["test_verify"] = challenge
        assert service.verify_challenge("test_verify", "42")
    
    def test_verify_removes_challenge(self):
        service = CaptchaService()
        challenge = service.generate_challenge()
        cid = challenge.challenge_id
        service.verify_challenge(cid, "wrong")
        assert cid not in service.challenges
    
    def test_get_stats(self):
        service = CaptchaService()
        service.generate_challenge()
        stats = service.get_stats()
        assert stats['active_challenges'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
