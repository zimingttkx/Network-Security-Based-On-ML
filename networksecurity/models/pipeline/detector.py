"""
统一检测器
整合所有算法的完整检测系统
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import time

from networksecurity.models.pipeline.preprocessor import UnifiedPreprocessor, OutputFormat
from networksecurity.models.pipeline.adapter import ModelAdapter, PipelineStage, StageResult

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """检测结果"""
    is_threat: bool
    threat_score: float
    threat_type: str
    action: str
    confidence: float
    detection_time_ms: float
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            'is_threat': self.is_threat,
            'threat_score': self.threat_score,
            'threat_type': self.threat_type,
            'action': self.action,
            'confidence': self.confidence,
            'detection_time_ms': self.detection_time_ms,
            'details': self.details
        }


class UnifiedDetector:
    """
    统一检测器
    
    整合多种检测算法:
    1. Kitsune - 无监督异常检测
    2. LUCID - DDoS分类
    3. Slips - 行为分析
    4. RL代理 - 智能决策
    
    支持多种组合模式:
    - 串行: 依次执行各检测器
    - 并行: 同时执行，投票决策
    - 级联: 异常检测 -> 分类 -> 决策
    """
    
    def __init__(self, mode: str = "cascade"):
        """
        Args:
            mode: 检测模式 ("serial", "parallel", "cascade")
        """
        self.mode = mode
        self.preprocessor = UnifiedPreprocessor()
        self.detectors: Dict[str, ModelAdapter] = {}
        self.rl_agent: Optional[ModelAdapter] = None
        
        # 配置
        self.threat_threshold = 0.5
        self.voting_strategy = "max"  # "max", "avg", "majority"
        
        # 统计
        self.total_processed = 0
        self.total_threats = 0
        self.detection_times: List[float] = []
    
    def add_detector(self, name: str, model: Any, model_type: str):
        """添加检测器"""
        adapter = ModelAdapter(model, model_type)
        self.detectors[name] = adapter
        logger.info(f"添加检测器: {name} ({model_type})")
    
    def set_rl_agent(self, agent: Any):
        """设置RL决策代理"""
        self.rl_agent = ModelAdapter(agent, 'rl')
    
    def detect(self, data: Any) -> DetectionResult:
        """执行检测"""
        start_time = time.time()
        self.total_processed += 1
        
        # 预处理
        processed = self.preprocessor.preprocess(data, OutputFormat.UNIFIED)
        features = processed.features
        
        # 根据模式执行检测
        if self.mode == "parallel":
            results = self._detect_parallel(features)
        elif self.mode == "cascade":
            results = self._detect_cascade(features, data)
        else:
            results = self._detect_serial(features)
        
        # 综合结果
        final_result = self._aggregate_results(results, features)
        
        detection_time = (time.time() - start_time) * 1000
        self.detection_times.append(detection_time)
        
        if final_result.is_threat:
            self.total_threats += 1
        
        final_result.detection_time_ms = detection_time
        return final_result
    
    def _detect_serial(self, features: np.ndarray) -> List[StageResult]:
        """串行检测"""
        results = []
        for name, adapter in self.detectors.items():
            try:
                output_format = self._get_output_format(adapter.model_type)
                processed = self.preprocessor.preprocess(features, output_format)
                result = adapter.process(processed.features)
                result.metadata['detector'] = name
                results.append(result)
            except Exception as e:
                logger.warning(f"检测器 {name} 失败: {e}")
        return results
    
    def _detect_parallel(self, features: np.ndarray) -> List[StageResult]:
        """并行检测 (实际为顺序执行，但结果独立)"""
        return self._detect_serial(features)
    
    def _detect_cascade(self, features: np.ndarray, raw_data: Any) -> List[StageResult]:
        """级联检测: 异常检测 -> 分类 -> 决策"""
        results = []
        
        # 阶段1: 异常检测 (Kitsune, Slips)
        anomaly_detectors = {k: v for k, v in self.detectors.items() 
                           if v.stage == PipelineStage.ANOMALY_DETECTION}
        
        is_anomaly = False
        anomaly_score = 0.0
        
        for name, adapter in anomaly_detectors.items():
            try:
                output_format = self._get_output_format(adapter.model_type)
                processed = self.preprocessor.preprocess(features, output_format)
                result = adapter.process(processed.features)
                result.metadata['detector'] = name
                results.append(result)
                
                if result.is_threat:
                    is_anomaly = True
                anomaly_score = max(anomaly_score, result.score)
            except Exception as e:
                logger.warning(f"异常检测器 {name} 失败: {e}")
        
        # 阶段2: 分类 (LUCID, ML/DL)
        if is_anomaly or anomaly_score > 0.3:
            classifiers = {k: v for k, v in self.detectors.items() 
                         if v.stage == PipelineStage.CLASSIFICATION}
            
            for name, adapter in classifiers.items():
                try:
                    output_format = self._get_output_format(adapter.model_type)
                    processed = self.preprocessor.preprocess(features, output_format)
                    result = adapter.process(processed.features)
                    result.metadata['detector'] = name
                    results.append(result)
                except Exception as e:
                    logger.warning(f"分类器 {name} 失败: {e}")
        
        # 阶段3: RL决策
        if self.rl_agent and self.rl_agent.is_ready():
            try:
                rl_features = self.preprocessor.preprocess(features, OutputFormat.RL_STATE).features
                # 添加检测结果到状态
                rl_features = self._enrich_rl_state(rl_features, results)
                result = self.rl_agent.process(rl_features)
                result.metadata['detector'] = 'rl_agent'
                results.append(result)
            except Exception as e:
                logger.warning(f"RL代理失败: {e}")
        
        return results
    
    def _get_output_format(self, model_type: str) -> OutputFormat:
        """获取模型所需的输出格式"""
        format_map = {
            'kitsune': OutputFormat.KITSUNE,
            'lucid': OutputFormat.LUCID,
            'slips': OutputFormat.SLIPS,
            'rl': OutputFormat.RL_STATE
        }
        return format_map.get(model_type, OutputFormat.UNIFIED)
    
    def _enrich_rl_state(self, state: np.ndarray, results: List[StageResult]) -> np.ndarray:
        """用检测结果丰富RL状态"""
        enriched = state.copy()
        
        # 添加威胁分数
        threat_scores = [r.score for r in results]
        if threat_scores:
            enriched[7] = max(threat_scores)  # threat_score
            enriched[8] = np.mean(threat_scores)  # anomaly_score
        
        return enriched
    
    def _aggregate_results(self, results: List[StageResult], 
                          features: np.ndarray) -> DetectionResult:
        """聚合检测结果"""
        if not results:
            return DetectionResult(
                is_threat=False, threat_score=0.0, threat_type="normal",
                action="ALLOW", confidence=1.0, detection_time_ms=0.0, details={}
            )
        
        # 收集分数
        scores = [r.score for r in results if r.score > 0]
        threats = [r for r in results if r.is_threat]
        
        # 计算综合分数
        if self.voting_strategy == "max":
            threat_score = max(scores) if scores else 0.0
        elif self.voting_strategy == "avg":
            threat_score = np.mean(scores) if scores else 0.0
        else:  # majority
            threat_score = len(threats) / len(results) if results else 0.0
        
        is_threat = threat_score >= self.threat_threshold
        
        # 确定威胁类型
        threat_type = "normal"
        if is_threat:
            for r in results:
                if r.is_threat and 'threat_types' in r.metadata:
                    threat_type = r.metadata['threat_types'][0]
                    break
            if threat_type == "normal":
                threat_type = "anomaly"
        
        # 确定动作
        action = "ALLOW"
        rl_results = [r for r in results if r.stage == PipelineStage.RL_DECISION]
        if rl_results:
            action = rl_results[-1].metadata.get('action_name', 'ALLOW')
        elif is_threat:
            action = "BLOCK" if threat_score > 0.7 else "ALERT"
        
        # 计算置信度
        confidence = 1.0 - abs(threat_score - 0.5) * 2 if 0.3 < threat_score < 0.7 else 0.9
        
        return DetectionResult(
            is_threat=is_threat,
            threat_score=threat_score,
            threat_type=threat_type,
            action=action,
            confidence=confidence,
            detection_time_ms=0.0,
            details={
                'detectors_used': [r.metadata.get('detector', 'unknown') for r in results],
                'individual_scores': {r.metadata.get('detector', f'det_{i}'): r.score 
                                     for i, r in enumerate(results)}
            }
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_processed': self.total_processed,
            'total_threats': self.total_threats,
            'threat_rate': self.total_threats / max(1, self.total_processed),
            'avg_detection_time_ms': np.mean(self.detection_times) if self.detection_times else 0,
            'detectors': list(self.detectors.keys()),
            'mode': self.mode
        }
