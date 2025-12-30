"""
DL模型基类
定义深度学习模型的抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type, Union
from enum import Enum
import numpy as np
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)


class DLModelType(str, Enum):
    """DL模型类型"""
    DNN = "dnn"
    CNN = "cnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"
    VAE = "vae"


@dataclass
class DLModelInfo:
    """DL模型信息"""
    name: str
    type: DLModelType
    description: str
    supports_proba: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'supports_proba': self.supports_proba,
            'default_params': self.default_params
        }


@dataclass 
class DLTrainingResult:
    """DL训练结果"""
    success: bool
    model_name: str
    train_time: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=dict)
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'model_name': self.model_name,
            'train_time': self.train_time,
            'final_loss': self.final_loss,
            'final_accuracy': self.final_accuracy,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'error': self.error
        }


class DLModelBase(ABC):
    """深度学习模型基类"""
    
    def __init__(self, input_dim: int = None, num_classes: int = 2, **kwargs):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.params = kwargs
        self.is_fitted = False
        self.history = None
        self._training_time = 0.0
    
    @property
    @abstractmethod
    def info(self) -> DLModelInfo:
        """返回模型信息"""
        pass
    
    @abstractmethod
    def _build_model(self) -> Any:
        """构建模型架构"""
        pass
    
    def _check_tensorflow(self):
        """检查TensorFlow是否可用"""
        try:
            import tensorflow as tf
            return tf
        except ImportError:
            raise ImportError("请安装tensorflow: pip install tensorflow")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> DLTrainingResult:
        """训练模型"""
        start_time = time.time()
        
        try:
            tf = self._check_tensorflow()
            
            X = np.array(X) if not isinstance(X, np.ndarray) else X
            y = np.array(y) if not isinstance(y, np.ndarray) else y
            
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            
            self.model = self._build_model()
            
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
                y_val = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
                validation_data = (X_val, y_val)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
            
            self.history = self.model.fit(
                X, y, epochs=epochs, batch_size=batch_size,
                validation_data=validation_data, callbacks=callbacks, verbose=verbose
            )
            
            self.is_fitted = True
            self._training_time = time.time() - start_time
            
            history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
            
            return DLTrainingResult(
                success=True,
                model_name=self.info.name,
                train_time=self._training_time,
                history=history_dict,
                final_loss=float(history_dict.get('loss', [0])[-1]),
                final_accuracy=float(history_dict.get('accuracy', [0])[-1]),
                val_loss=float(history_dict.get('val_loss', [0])[-1]) if 'val_loss' in history_dict else 0,
                val_accuracy=float(history_dict.get('val_accuracy', [0])[-1]) if 'val_accuracy' in history_dict else 0
            )
        except Exception as e:
            logger.error(f"DL训练失败: {e}")
            return DLTrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten() if proba.shape[1] == 1 else np.argmax(proba, axis=1)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return self.model.predict(X, verbose=0)
    
    def save(self, path: str) -> bool:
        try:
            if self.model:
                self.model.save(path)
                return True
            return False
        except Exception as e:
            logger.error(f"保存DL模型失败: {e}")
            return False
    
    def load(self, path: str) -> bool:
        try:
            tf = self._check_tensorflow()
            self.model = tf.keras.models.load_model(path)
            self.is_fitted = True
            return True
        except Exception as e:
            logger.error(f"加载DL模型失败: {e}")
            return False


class DLModelRegistry:
    """DL模型注册表"""
    
    _models: Dict[str, Type[DLModelBase]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(model_class: Type[DLModelBase]):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[DLModelBase]]:
        return cls._models.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[DLModelBase]:
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
            except:
                pass
        return result
