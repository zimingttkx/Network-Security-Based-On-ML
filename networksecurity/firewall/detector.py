"""
威胁检测器
集成多种模型进行威胁检测
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """威胁等级"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """响应动作"""
    ALLOW = "allow"
    LOG = "log"
    ALERT = "alert"
    CHALLENGE = "challenge"
    BLOCK = "block"


@dataclass
class DetectionResult:
    """检测结果"""
    is_threat: bool
    threat_level: ThreatLevel
    confidence: float
    action: ActionType
    model_name: str
    threat_type: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    detection_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_threat': self.is_threat,
            'threat_level': self.threat_level.value,
            'confidence': self.confidence,
            'action': self.action.value,
            'model_name': self.model_name,
            'threat_type': self.threat_type,
            'details': self.details,
            'detection_time': self.detection_time
        }


class ThreatDetector:
    """威胁检测器"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.default_model: Optional[str] = None
        self.thresholds = {
            ThreatLevel.SAFE: 0.2,
            ThreatLevel.LOW: 0.4,
            ThreatLevel.MEDIUM: 0.6,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 0.95
        }
    
    def register_model(self, name: str, model: Any, set_default: bool = False):
        """注册检测模型"""
        self.models[name] = model
        if set_default or self.default_model is None:
            self.default_model = name
        logger.info(f"注册模型: {name}")
    
    def unregister_model(self, name: str):
        """注销模型"""
        if name in self.models:
            del self.models[name]
            if self.default_model == name:
                self.default_model = next(iter(self.models), None)
    
    def detect(self, features: Union[np.ndarray, pd.DataFrame, Dict], 
               model_name: Optional[str] = None) -> DetectionResult:
        """执行威胁检测"""
        start_time = time.time()
        model_name = model_name or self.default_model
        
        if not model_name or model_name not in self.models:
            return self._fallback_detection(features, start_time)
        
        try:
            model = self.models[model_name]
            features_arr = self._prepare_features(features)
            
            # 获取预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_arr)
                confidence = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            else:
                pred = model.predict(features_arr)
                confidence = float(pred[0]) if isinstance(pred[0], (int, float)) else 0.5
            
            is_threat = confidence > self.thresholds[ThreatLevel.SAFE]
            threat_level = self._get_threat_level(confidence)
            action = self._get_action(threat_level)
            
            return DetectionResult(
                is_threat=is_threat,
                threat_level=threat_level,
                confidence=confidence,
                action=action,
                model_name=model_name,
                detection_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return self._fallback_detection(features, start_time)
    
    def detect_batch(self, features_list: List[Union[np.ndarray, Dict]], 
                     model_name: Optional[str] = None) -> List[DetectionResult]:
        """批量检测"""
        return [self.detect(f, model_name) for f in features_list]
    
    def _prepare_features(self, features: Union[np.ndarray, pd.DataFrame, Dict]) -> np.ndarray:
        """准备特征数据"""
        if isinstance(features, dict):
            return np.array(list(features.values())).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            return features.values
        elif isinstance(features, np.ndarray):
            return features.reshape(1, -1) if features.ndim == 1 else features
        return np.array(features).reshape(1, -1)
    
    def _get_threat_level(self, confidence: float) -> ThreatLevel:
        """根据置信度获取威胁等级"""
        if confidence >= self.thresholds[ThreatLevel.CRITICAL]:
            return ThreatLevel.CRITICAL
        elif confidence >= self.thresholds[ThreatLevel.HIGH]:
            return ThreatLevel.HIGH
        elif confidence >= self.thresholds[ThreatLevel.MEDIUM]:
            return ThreatLevel.MEDIUM
        elif confidence >= self.thresholds[ThreatLevel.LOW]:
            return ThreatLevel.LOW
        return ThreatLevel.SAFE
    
    def _get_action(self, threat_level: ThreatLevel) -> ActionType:
        """根据威胁等级获取响应动作"""
        action_map = {
            ThreatLevel.SAFE: ActionType.ALLOW,
            ThreatLevel.LOW: ActionType.LOG,
            ThreatLevel.MEDIUM: ActionType.ALERT,
            ThreatLevel.HIGH: ActionType.CHALLENGE,
            ThreatLevel.CRITICAL: ActionType.BLOCK
        }
        return action_map.get(threat_level, ActionType.LOG)
    
    def _fallback_detection(self, features, start_time: float) -> DetectionResult:
        """回退检测（无模型时）"""
        return DetectionResult(
            is_threat=False,
            threat_level=ThreatLevel.SAFE,
            confidence=0.0,
            action=ActionType.ALLOW,
            model_name="fallback",
            details={'reason': 'no_model_available'},
            detection_time=time.time() - start_time
        )
    
    def list_models(self) -> List[str]:
        """列出所有注册的模型"""
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'models': self.list_models(),
            'default_model': self.default_model,
            'thresholds': {k.value: v for k, v in self.thresholds.items()}
        }
