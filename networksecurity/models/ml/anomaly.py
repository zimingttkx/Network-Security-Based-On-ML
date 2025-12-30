"""
异常检测模型实现
用于无监督异常检测
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import pandas as pd
import logging

from networksecurity.models.ml.base import MLModelBase, ModelRegistry, ModelInfo, ModelType, TrainingResult

logger = logging.getLogger(__name__)


@ModelRegistry.register("isolation_forest")
class IsolationForestDetector(MLModelBase):
    """孤立森林异常检测器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Isolation Forest",
            type=ModelType.ANOMALY,
            description="基于孤立森林的异常检测",
            supports_proba=False,
            default_params={'n_estimators': 100, 'contamination': 'auto'}
        )
    
    def _create_model(self):
        from sklearn.ensemble import IsolationForest
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', -1)
        return IsolationForest(**params)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None, **kwargs) -> TrainingResult:
        """异常检测器只需要X"""
        import time
        start_time = time.time()
        try:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            self.model = self._create_model()
            self.model.fit(X)
            self.is_fitted = True
            self._training_time = time.time() - start_time
            return TrainingResult(success=True, model_name=self.info.name, train_time=self._training_time)
        except Exception as e:
            return TrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """返回1(正常)或-1(异常)"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """返回异常分数"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.score_samples(X)


@ModelRegistry.register("one_class_svm")
class OneClassSVMDetector(MLModelBase):
    """单类SVM异常检测器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="One-Class SVM",
            type=ModelType.ANOMALY,
            description="基于单类SVM的异常检测",
            supports_proba=False,
            default_params={'kernel': 'rbf', 'nu': 0.1}
        )
    
    def _create_model(self):
        from sklearn.svm import OneClassSVM
        params = {**self.info.default_params, **self.params}
        return OneClassSVM(**params)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None, **kwargs) -> TrainingResult:
        import time
        start_time = time.time()
        try:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            self.model = self._create_model()
            self.model.fit(X)
            self.is_fitted = True
            self._training_time = time.time() - start_time
            return TrainingResult(success=True, model_name=self.info.name, train_time=self._training_time)
        except Exception as e:
            return TrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)


@ModelRegistry.register("lof")
class LOFDetector(MLModelBase):
    """局部异常因子检测器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Local Outlier Factor",
            type=ModelType.ANOMALY,
            description="基于局部异常因子的异常检测",
            supports_proba=False,
            default_params={'n_neighbors': 20, 'contamination': 'auto'}
        )
    
    def _create_model(self):
        from sklearn.neighbors import LocalOutlierFactor
        params = {**self.info.default_params, **self.params}
        params['novelty'] = True  # 允许对新数据预测
        params.setdefault('n_jobs', -1)
        return LocalOutlierFactor(**params)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None, **kwargs) -> TrainingResult:
        import time
        start_time = time.time()
        try:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            self.model = self._create_model()
            self.model.fit(X)
            self.is_fitted = True
            self._training_time = time.time() - start_time
            return TrainingResult(success=True, model_name=self.info.name, train_time=self._training_time)
        except Exception as e:
            return TrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
