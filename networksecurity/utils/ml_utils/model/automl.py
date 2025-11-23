"""
AutoML模块 - 自动化机器学习
使用Optuna进行自动超参数优化
"""
import sys
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class AutoMLOptimizer:
    """自动化机器学习优化器"""

    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = 3600,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        初始化AutoML优化器

        Args:
            n_trials: Optuna优化试验次数
            timeout: 超时时间（秒）
            cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = {}
        self.best_score = 0.0
        self.study = None

    def _objective_rf(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """RandomForest目标函数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': self.random_state
        }

        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1', n_jobs=-1).mean()
        return score

    def _objective_gb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """GradientBoosting目标函数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': self.random_state
        }

        model = GradientBoostingClassifier(**params)
        score = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1', n_jobs=-1).mean()
        return score

    def _objective_xgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """XGBoost目标函数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': self.random_state,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }

        model = xgb.XGBClassifier(**params)
        score = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1', n_jobs=-1).mean()
        return score

    def _objective_lgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """LightGBM目标函数"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': self.random_state,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        score = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1', n_jobs=-1).mean()
        return score

    def _objective_cb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """CatBoost目标函数"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': self.random_state,
            'verbose': False,
            'allow_writing_files': False
        }

        model = cb.CatBoostClassifier(**params)
        score = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1', n_jobs=-1).mean()
        return score

    def optimize(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        """
        优化指定模型

        Args:
            model_type: 模型类型 (rf, gb, xgb, lgb, cb)
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            最佳参数和得分
        """
        try:
            logging.info(f"开始AutoML优化: {model_type}")

            # 选择目标函数
            objective_map = {
                'rf': self._objective_rf,
                'gb': self._objective_gb,
                'xgb': self._objective_xgb,
                'lgb': self._objective_lgb,
                'cb': self._objective_cb
            }

            if model_type not in objective_map:
                raise ValueError(f"不支持的模型类型: {model_type}")

            objective_func = objective_map[model_type]

            # 创建Optuna study
            self.study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )

            # 优化
            self.study.optimize(
                lambda trial: objective_func(trial, X_train, y_train),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )

            self.best_params = self.study.best_params
            self.best_score = self.study.best_value

            logging.info(f"AutoML优化完成: {model_type}")
            logging.info(f"最佳得分: {self.best_score:.4f}")
            logging.info(f"最佳参数: {self.best_params}")

            return self.best_params, self.best_score

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_optimization_history(self) -> Dict[str, Any]:
        """获取优化历史"""
        if self.study is None:
            return {}

        return {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                }
                for trial in self.study.trials
            ]
        }
