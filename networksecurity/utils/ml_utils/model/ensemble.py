"""
集成学习模块
实现Voting、Stacking和Blending策略
"""
import sys
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class EnsembleBuilder:
    """集成学习构建器"""

    def __init__(self, random_state: int = 42):
        """
        初始化集成学习构建器

        Args:
            random_state: 随机种子
        """
        self.random_state = random_state

    def create_voting_ensemble(
        self,
        estimators: List[tuple],
        voting: str = 'soft',
        weights: List[float] = None
    ) -> VotingClassifier:
        """
        创建投票集成模型

        Args:
            estimators: 基础模型列表，格式: [('name', model), ...]
            voting: 投票方式 ('hard' 或 'soft')
            weights: 模型权重

        Returns:
            投票集成分类器
        """
        try:
            logging.info(f"创建投票集成模型，投票方式: {voting}")

            ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting,
                weights=weights,
                n_jobs=-1
            )

            logging.info(f"投票集成模型创建完成，包含 {len(estimators)} 个基础模型")
            return ensemble

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def create_stacking_ensemble(
        self,
        estimators: List[tuple],
        final_estimator=None,
        cv: int = 5
    ) -> StackingClassifier:
        """
        创建堆叠集成模型

        Args:
            estimators: 基础模型列表，格式: [('name', model), ...]
            final_estimator: 元学习器，默认为LogisticRegression
            cv: 交叉验证折数

        Returns:
            堆叠集成分类器
        """
        try:
            logging.info("创建堆叠集成模型")

            if final_estimator is None:
                final_estimator = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )

            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=cv,
                n_jobs=-1
            )

            logging.info(f"堆叠集成模型创建完成，包含 {len(estimators)} 个基础模型")
            return ensemble

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def evaluate_ensemble(
        self,
        ensemble,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        评估集成模型

        Args:
            ensemble: 集成模型
            X_train: 训练特征
            y_train: 训练标签
            cv: 交叉验证折数

        Returns:
            评估指标字典
        """
        try:
            logging.info("开始评估集成模型")

            # 交叉验证评分
            scoring = ['accuracy', 'precision', 'recall', 'f1']
            scores = {}

            for metric in scoring:
                cv_scores = cross_val_score(
                    ensemble, X_train, y_train,
                    cv=cv, scoring=metric, n_jobs=-1
                )
                scores[f'{metric}_mean'] = cv_scores.mean()
                scores[f'{metric}_std'] = cv_scores.std()

                logging.info(f"{metric}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            return scores

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def optimize_voting_weights(
        self,
        estimators: List[tuple],
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> List[float]:
        """
        优化投票集成的权重

        Args:
            estimators: 基础模型列表
            X_train: 训练特征
            y_train: 训练标签
            cv: 交叉验证折数

        Returns:
            优化后的权重列表
        """
        try:
            logging.info("开始优化投票权重")

            # 计算每个模型的交叉验证得分
            scores = []
            for name, estimator in estimators:
                cv_scores = cross_val_score(
                    estimator, X_train, y_train,
                    cv=cv, scoring='f1', n_jobs=-1
                )
                score = cv_scores.mean()
                scores.append(score)
                logging.info(f"{name} F1得分: {score:.4f}")

            # 归一化得分作为权重
            scores = np.array(scores)
            weights = scores / scores.sum()

            logging.info(f"优化后的权重: {weights.tolist()}")
            return weights.tolist()

        except Exception as e:
            raise NetworkSecurityException(e, sys)


class BlendingEnsemble:
    """Blending集成模型"""

    def __init__(self, base_models: List[Any], meta_model: Any = None, test_size: float = 0.2):
        """
        初始化Blending集成

        Args:
            base_models: 基础模型列表
            meta_model: 元学习器
            test_size: 用于训练元学习器的数据比例
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.test_size = test_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练Blending集成模型

        Args:
            X: 训练特征
            y: 训练标签
        """
        try:
            from sklearn.model_selection import train_test_split

            # 分割数据
            X_train, X_blend, y_train, y_blend = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )

            # 训练基础模型
            blend_features = np.zeros((X_blend.shape[0], len(self.base_models)))

            for i, model in enumerate(self.base_models):
                logging.info(f"训练基础模型 {i+1}/{len(self.base_models)}")
                model.fit(X_train, y_train)

                # 生成blend特征
                if hasattr(model, 'predict_proba'):
                    blend_features[:, i] = model.predict_proba(X_blend)[:, 1]
                else:
                    blend_features[:, i] = model.predict(X_blend)

            # 训练元学习器
            logging.info("训练元学习器")
            self.meta_model.fit(blend_features, y_blend)

            # 重新在全部数据上训练基础模型
            for model in self.base_models:
                model.fit(X, y)

            logging.info("Blending集成训练完成")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        try:
            # 生成blend特征
            blend_features = np.zeros((X.shape[0], len(self.base_models)))

            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba'):
                    blend_features[:, i] = model.predict_proba(X)[:, 1]
                else:
                    blend_features[:, i] = model.predict(X)

            # 元学习器预测
            return self.meta_model.predict(blend_features)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征数据

        Returns:
            预测概率
        """
        try:
            # 生成blend特征
            blend_features = np.zeros((X.shape[0], len(self.base_models)))

            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba'):
                    blend_features[:, i] = model.predict_proba(X)[:, 1]
                else:
                    blend_features[:, i] = model.predict(X)

            # 元学习器预测概率
            if hasattr(self.meta_model, 'predict_proba'):
                return self.meta_model.predict_proba(blend_features)
            else:
                # 如果元学习器不支持概率预测，返回预测结果
                predictions = self.meta_model.predict(blend_features)
                proba = np.zeros((len(predictions), 2))
                proba[np.arange(len(predictions)), predictions.astype(int)] = 1
                return proba

        except Exception as e:
            raise NetworkSecurityException(e, sys)
