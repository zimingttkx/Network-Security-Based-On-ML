"""
ML分类器实现
包含多种机器学习分类算法
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from networksecurity.models.ml.base import MLModelBase, ModelRegistry, ModelInfo, ModelType

logger = logging.getLogger(__name__)


@ModelRegistry.register("xgboost")
class XGBoostClassifier(MLModelBase):
    """XGBoost分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="XGBoost",
            type=ModelType.ENSEMBLE,
            description="极端梯度提升分类器，高效且准确",
            default_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        )
    
    def _create_model(self):
        try:
            from xgboost import XGBClassifier
            params = {**self.info.default_params, **self.params}
            params.setdefault('random_state', 42)
            params.setdefault('use_label_encoder', False)
            params.setdefault('eval_metric', 'logloss')
            return XGBClassifier(**params)
        except ImportError:
            raise ImportError("请安装xgboost: pip install xgboost")


@ModelRegistry.register("lightgbm")
class LightGBMClassifier(MLModelBase):
    """LightGBM分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="LightGBM",
            type=ModelType.ENSEMBLE,
            description="轻量级梯度提升机，训练速度快",
            default_params={'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1}
        )
    
    def _create_model(self):
        try:
            from lightgbm import LGBMClassifier
            params = {**self.info.default_params, **self.params}
            params.setdefault('random_state', 42)
            params.setdefault('verbose', -1)
            return LGBMClassifier(**params)
        except ImportError:
            raise ImportError("请安装lightgbm: pip install lightgbm")


@ModelRegistry.register("catboost")
class CatBoostClassifier(MLModelBase):
    """CatBoost分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="CatBoost",
            type=ModelType.ENSEMBLE,
            description="支持类别特征的梯度提升",
            default_params={'iterations': 100, 'depth': 6, 'learning_rate': 0.1}
        )
    
    def _create_model(self):
        try:
            from catboost import CatBoostClassifier as CBC
            params = {**self.info.default_params, **self.params}
            params.setdefault('random_state', 42)
            params.setdefault('verbose', False)
            return CBC(**params)
        except ImportError:
            raise ImportError("请安装catboost: pip install catboost")


@ModelRegistry.register("random_forest")
class RandomForestClassifier(MLModelBase):
    """随机森林分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Random Forest",
            type=ModelType.ENSEMBLE,
            description="随机森林集成分类器",
            default_params={'n_estimators': 100, 'max_depth': None}
        )
    
    def _create_model(self):
        from sklearn.ensemble import RandomForestClassifier as RFC
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', -1)
        return RFC(**params)


@ModelRegistry.register("gradient_boosting")
class GradientBoostingClassifier(MLModelBase):
    """梯度提升分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Gradient Boosting",
            type=ModelType.ENSEMBLE,
            description="梯度提升分类器",
            default_params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
        )
    
    def _create_model(self):
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        return GBC(**params)


@ModelRegistry.register("adaboost")
class AdaBoostClassifier(MLModelBase):
    """AdaBoost分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="AdaBoost",
            type=ModelType.ENSEMBLE,
            description="自适应提升分类器",
            default_params={'n_estimators': 50, 'learning_rate': 1.0}
        )
    
    def _create_model(self):
        from sklearn.ensemble import AdaBoostClassifier as ABC
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        return ABC(**params)


@ModelRegistry.register("extra_trees")
class ExtraTreesClassifier(MLModelBase):
    """极端随机树分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Extra Trees",
            type=ModelType.ENSEMBLE,
            description="极端随机树分类器",
            default_params={'n_estimators': 100, 'max_depth': None}
        )
    
    def _create_model(self):
        from sklearn.ensemble import ExtraTreesClassifier as ETC
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', -1)
        return ETC(**params)


@ModelRegistry.register("svm")
class SVMClassifier(MLModelBase):
    """支持向量机分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="SVM",
            type=ModelType.SVM,
            description="支持向量机分类器",
            default_params={'C': 1.0, 'kernel': 'rbf'}
        )
    
    def _create_model(self):
        from sklearn.svm import SVC
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        params.setdefault('probability', True)
        return SVC(**params)


@ModelRegistry.register("logistic_regression")
class LogisticRegressionClassifier(MLModelBase):
    """逻辑回归分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Logistic Regression",
            type=ModelType.LINEAR,
            description="逻辑回归分类器",
            default_params={'C': 1.0, 'max_iter': 1000}
        )
    
    def _create_model(self):
        from sklearn.linear_model import LogisticRegression as LR
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', -1)
        return LR(**params)


@ModelRegistry.register("knn")
class KNNClassifier(MLModelBase):
    """K近邻分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="KNN",
            type=ModelType.NEIGHBORS,
            description="K近邻分类器",
            default_params={'n_neighbors': 5}
        )
    
    def _create_model(self):
        from sklearn.neighbors import KNeighborsClassifier as KNC
        params = {**self.info.default_params, **self.params}
        params.setdefault('n_jobs', -1)
        return KNC(**params)


@ModelRegistry.register("naive_bayes")
class NaiveBayesClassifier(MLModelBase):
    """朴素贝叶斯分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Naive Bayes",
            type=ModelType.NAIVE_BAYES,
            description="高斯朴素贝叶斯分类器",
            default_params={}
        )
    
    def _create_model(self):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**self.params)


@ModelRegistry.register("decision_tree")
class DecisionTreeClassifier(MLModelBase):
    """决策树分类器"""
    
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Decision Tree",
            type=ModelType.TREE,
            description="决策树分类器",
            default_params={'max_depth': None}
        )
    
    def _create_model(self):
        from sklearn.tree import DecisionTreeClassifier as DTC
        params = {**self.info.default_params, **self.params}
        params.setdefault('random_state', 42)
        return DTC(**params)
