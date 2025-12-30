"""
ML模型基类和注册表
定义机器学习模型的抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type, Union
from enum import Enum
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """模型类型枚举"""
    ENSEMBLE = "ensemble"
    LINEAR = "linear"
    TREE = "tree"
    SVM = "svm"
    NEIGHBORS = "neighbors"
    NAIVE_BAYES = "naive_bayes"
    ANOMALY = "anomaly"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    type: ModelType
    description: str
    supports_proba: bool = True
    supports_multiclass: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'supports_proba': self.supports_proba,
            'supports_multiclass': self.supports_multiclass,
            'default_params': self.default_params
        }


@dataclass
class TrainingResult:
    """训练结果"""
    success: bool
    model_name: str
    train_time: float = 0.0
    train_score: float = 0.0
    val_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'model_name': self.model_name,
            'train_time': self.train_time,
            'train_score': self.train_score,
            'val_score': self.val_score,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'error': self.error
        }


class MLModelBase(ABC):
    """机器学习模型基类"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs
        self.is_fitted = False
        self.feature_names: List[str] = []
        self._training_time = 0.0
    
    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """返回模型信息"""
        pass
    
    @abstractmethod
    def _create_model(self) -> Any:
        """创建底层模型实例"""
        pass
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None) -> TrainingResult:
        """训练模型"""
        start_time = time.time()
        
        try:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            
            self.model = self._create_model()
            self.model.fit(X, y)
            self.is_fitted = True
            self._training_time = time.time() - start_time
            
            train_score = self.model.score(X, y)
            val_score = self.model.score(X_val, y_val) if X_val is not None else 0.0
            
            feature_importance = self._get_feature_importance()
            
            return TrainingResult(
                success=True,
                model_name=self.info.name,
                train_time=self._training_time,
                train_score=train_score,
                val_score=val_score,
                feature_importance=feature_importance
            )
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return TrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        if not self.info.supports_proba:
            raise ValueError(f"{self.info.name} 不支持概率预测")
        return self.model.predict_proba(X)
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """评估模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.score(X, y)
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if not self.is_fitted or not self.feature_names:
            return None
        
        importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
            if len(importance) != len(self.feature_names):
                return None
        
        if importance is not None:
            return dict(zip(self.feature_names, importance.tolist()))
        return None
    
    def save(self, path: str) -> bool:
        """保存模型"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({'model': self.model, 'params': self.params, 
                           'feature_names': self.feature_names, 'is_fitted': self.is_fitted}, f)
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """加载模型"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.params = data.get('params', {})
            self.feature_names = data.get('feature_names', [])
            self.is_fitted = data.get('is_fitted', True)
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False


class ModelRegistry:
    """模型注册表"""
    
    _models: Dict[str, Type[MLModelBase]] = {}
    
    @classmethod
    def register(cls, name: str):
        """注册模型的装饰器"""
        def decorator(model_class: Type[MLModelBase]):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[MLModelBase]]:
        return cls._models.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[MLModelBase]:
        model_class = cls.get(name)
        return model_class(**kwargs) if model_class else None
    
    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())
    
    @classmethod
    def get_all_info(cls) -> List[Dict[str, Any]]:
        result = []
        for name, model_class in cls._models.items():
            try:
                instance = model_class()
                result.append(instance.info.to_dict())
            except Exception as e:
                logger.warning(f"获取模型 {name} 信息失败: {e}")
        return result
