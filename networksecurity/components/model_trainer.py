# networksecurity/components/model_trainer.py

import os
import sys
import logging
from functools import reduce
from operator import mul

import joblib

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, load_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from networksecurity.utils.ml_utils.model.estimator import NetworkModel


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def _count_combinations(self, param_grid):
        if not param_grid:
            return 1
        param_lengths = [len(v) if isinstance(v, list) else 1 for v in param_grid.values()]
        return reduce(mul, param_lengths, 1)

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'LogisticRegression': LogisticRegression(),
                'SVC': SVC(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'GaussianNB': GaussianNB(),
                'XGBoost': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            }
            params = {
                'RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [10, 20]},
                'GradientBoostingClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
                'AdaBoostClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]},
                'LogisticRegression': {'C': [0.1, 1], 'solver': ['liblinear']},
                'SVC': {'C': [1, 10], 'kernel': ['rbf']},
                'KNeighborsClassifier': {'n_neighbors': [3, 5]},
                'GaussianNB': {'var_smoothing': [1e-9]},
                'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
            }

            total_fits = sum(self._count_combinations(params.get(name, {})) * 3 for name in models)  # cv=3
            fits_done = 0
            logging.info(f"[PROGRESS]0/{total_fits}")

            model_report, best_estimators, best_params_report = {}, {}, {}

            for model_name, model in models.items():
                logging.info(f"====== 开始训练模型: {model_name} ======")
                param_grid = params.get(model_name, {})

                # --- 【重大修改】将 verbose 从 1 改为 3 ---
                # 这会让 GridSearchCV 打印出每一次拟合的详细日志
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=3)

                grid_search.fit(x_train, y_train)

                model_report[model_name] = grid_search.best_score_
                best_estimators[model_name] = grid_search.best_estimator_
                best_params_report[model_name] = grid_search.best_params_

                fits_this_round = self._count_combinations(param_grid) * 3
                fits_done += fits_this_round
                logging.info(f"[PROGRESS]{fits_done}/{total_fits}")
                logging.info(f"====== 模型 {model_name} 训练完成. ======")

            best_model_name = max(model_report, key=model_report.get)
            best_model = best_estimators[best_model_name]
            best_model_score = model_report[best_model_name]

            logging.info(f"==> 最佳模型: '{best_model_name}' | 交叉验证得分: {best_model_score:.4f}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"所有模型性能均未达到预期得分 {self.model_trainer_config.expected_accuracy}")

            # 在完整训练集上评估最佳模型
            y_train_pred = best_model.predict(x_train)
            train_metric = get_classification_metric(y_true=y_train, y_pred=y_train_pred)

            # 在测试集上评估
            y_test_pred = best_model.predict(x_test)
            test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"训练集指标: {train_metric}")
            logging.info(f"测试集指标: {test_metric}")

            # 保存最终模型 (预处理器 + 模型)
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            # 保存到本次运行的 artifact 目录
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            # 也保存一份到固定的 models 目录，供预测 API 使用
            final_model_dir = "models"
            os.makedirs(final_model_dir, exist_ok=True)
            save_object(os.path.join(final_model_dir, "model.pkl"), best_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("开始加载转换后的数据...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info("数据加载完毕，即将开始模型训练...")
            return self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e