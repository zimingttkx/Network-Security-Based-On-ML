"""
模型适配器
为各算法提供统一的接口
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """流水线阶段"""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    RL_DECISION = "rl_decision"


@dataclass
class StageResult:
    """阶段结果"""
    stage: PipelineStage
    output: Any
    score: float = 0.0
    is_threat: bool = False
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelAdapter:
    """
    模型适配器
    
    为不同算法提供统一接口，支持:
    - Kitsune (异常检测)
    - LUCID (DDoS分类)
    - Slips (行为分析)
    - RL代理 (决策)
    """
    
    def __init__(self, model: Any, model_type: str):
        """
        Args:
            model: 模型实例
            model_type: 模型类型 ("kitsune", "lucid", "slips", "rl")
        """
        self.model = model
        self.model_type = model_type
        self.stage = self._get_stage()
    
    def _get_stage(self) -> PipelineStage:
        """获取模型所属阶段"""
        stage_map = {
            'kitsune': PipelineStage.ANOMALY_DETECTION,
            'lucid': PipelineStage.CLASSIFICATION,
            'slips': PipelineStage.ANOMALY_DETECTION,
            'rl': PipelineStage.RL_DECISION,
            'ml': PipelineStage.CLASSIFICATION,
            'dl': PipelineStage.CLASSIFICATION
        }
        return stage_map.get(self.model_type, PipelineStage.CLASSIFICATION)
    
    def process(self, data: np.ndarray) -> StageResult:
        """处理数据"""
        if self.model_type == 'kitsune':
            return self._process_kitsune(data)
        elif self.model_type == 'lucid':
            return self._process_lucid(data)
        elif self.model_type == 'slips':
            return self._process_slips(data)
        elif self.model_type == 'rl':
            return self._process_rl(data)
        else:
            return self._process_generic(data)
    
    def _process_kitsune(self, data: np.ndarray) -> StageResult:
        """处理Kitsune"""
        result = self.model.process(data)
        return StageResult(
            stage=self.stage,
            output=result,
            score=result.rmse if hasattr(result, 'rmse') else 0.0,
            is_threat=result.is_anomaly if hasattr(result, 'is_anomaly') else False,
            metadata={'threshold': getattr(result, 'threshold', None)}
        )
    
    def _process_lucid(self, data: np.ndarray) -> StageResult:
        """处理LUCID"""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(data.reshape(1, *data.shape))[0]
            is_ddos = proba[1] > 0.5
            score = proba[1]
        else:
            pred = self.model.predict(data.reshape(1, -1))[0]
            is_ddos = pred == 1
            score = float(pred)
        
        return StageResult(
            stage=self.stage,
            output={'prediction': int(is_ddos), 'probability': score},
            score=score,
            is_threat=is_ddos
        )
    
    def _process_slips(self, data: np.ndarray) -> StageResult:
        """处理Slips"""
        if isinstance(data, np.ndarray):
            # 转换为字典格式
            flow_dict = {
                'src_ip': '0.0.0.0',
                'dst_ip': '0.0.0.0',
                'packet_size': int(data[0] * 1500) if len(data) > 0 else 0,
                'timestamp': 0.0
            }
        else:
            flow_dict = data
        
        result = self.model.process_packet(flow_dict)
        return StageResult(
            stage=self.stage,
            output=result.to_dict() if hasattr(result, 'to_dict') else result,
            score=result.threat_score if hasattr(result, 'threat_score') else 0.0,
            is_threat=result.is_threat if hasattr(result, 'is_threat') else False
        )
    
    def _process_rl(self, data: np.ndarray) -> StageResult:
        """处理RL代理"""
        action = self.model.select_action(data)
        return StageResult(
            stage=self.stage,
            output={'action': action},
            score=0.0,
            is_threat=action in [1, 6],  # BLOCK or QUARANTINE
            metadata={'action_name': ['ALLOW', 'BLOCK', 'ALERT', 'LOG', 'CHALLENGE', 'RATE_LIMIT', 'QUARANTINE'][action]}
        )
    
    def _process_generic(self, data: np.ndarray) -> StageResult:
        """处理通用模型"""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(data.reshape(1, -1))[0]
            pred = np.argmax(proba)
            score = proba[1] if len(proba) > 1 else proba[0]
        elif hasattr(self.model, 'predict'):
            pred = self.model.predict(data.reshape(1, -1))[0]
            score = float(pred)
        else:
            pred = 0
            score = 0.0
        
        return StageResult(
            stage=self.stage,
            output={'prediction': int(pred)},
            score=score,
            is_threat=pred == 1 or score > 0.5
        )
    
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        if hasattr(self.model, 'is_ready'):
            return self.model.is_ready()
        if hasattr(self.model, 'is_fitted'):
            return self.model.is_fitted
        if hasattr(self.model, 'is_trained'):
            return self.model.is_trained
        return True


class PipelineBuilder:
    """流水线构建器"""
    
    def __init__(self):
        self.stages: List[Tuple[PipelineStage, ModelAdapter]] = []
    
    def add_stage(self, adapter: ModelAdapter) -> 'PipelineBuilder':
        """添加阶段"""
        self.stages.append((adapter.stage, adapter))
        return self
    
    def build(self) -> 'Pipeline':
        """构建流水线"""
        return Pipeline(self.stages)


class Pipeline:
    """检测流水线"""
    
    def __init__(self, stages: List[Tuple[PipelineStage, ModelAdapter]]):
        self.stages = stages
    
    def process(self, data: np.ndarray) -> List[StageResult]:
        """处理数据通过所有阶段"""
        results = []
        current_data = data
        
        for stage_type, adapter in self.stages:
            result = adapter.process(current_data)
            results.append(result)
            
            # 如果检测到威胁且是阻断阶段，可以提前终止
            if result.is_threat and stage_type == PipelineStage.RL_DECISION:
                break
        
        return results
    
    def get_final_decision(self, results: List[StageResult]) -> Dict:
        """获取最终决策"""
        if not results:
            return {'is_threat': False, 'score': 0.0, 'action': 'ALLOW'}
        
        # 综合所有阶段结果
        threat_scores = [r.score for r in results if r.score > 0]
        is_threat = any(r.is_threat for r in results)
        
        final_score = max(threat_scores) if threat_scores else 0.0
        
        # 获取RL决策
        rl_results = [r for r in results if r.stage == PipelineStage.RL_DECISION]
        action = rl_results[-1].metadata.get('action_name', 'ALLOW') if rl_results else 'ALLOW'
        
        return {
            'is_threat': is_threat,
            'score': final_score,
            'action': action,
            'stages': len(results)
        }
