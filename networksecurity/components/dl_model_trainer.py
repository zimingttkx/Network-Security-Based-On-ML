# networksecurity/components/dl_model_trainer.py

import os
import sys
import logging
import numpy as np
from typing import Dict, Any

# 禁用TensorFlow的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, Model, regularizers, metrics
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    Model = None
    logging.warning(f"TensorFlow未安装，深度学习功能不可用: {e}")

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, load_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


class DLModelTrainer:
    """深度学习模型训练器"""

    def __init__(self, model_type='dnn', config=None,
                 model_trainer_config=None, data_transformation_artifact=None):
        """
        初始化深度学习模型训练器

        参数:
            model_type: 模型类型 ('dnn', 'cnn', 'lstm')
            config: 模型配置字典
            model_trainer_config: 训练配置对象（用于训练管道）
            data_transformation_artifact: 数据转换产物（用于训练管道）
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow未安装，无法使用深度学习功能")

            self.model_type = model_type
            self.config = config if config is not None else self.get_default_config(model_type)
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def create_dnn_model(self, input_dim: int, config: Dict[str, Any]) -> Model:
        """创建深度神经网络模型"""
        model = models.Sequential(name='DNN')

        # 输入层
        model.add(layers.Input(shape=(input_dim,)))

        # 隐藏层
        for i, units in enumerate(config['hidden_layers']):
            model.add(layers.Dense(
                units,
                activation=config['activation'],
                kernel_regularizer=regularizers.l2(config['l2_reg']),
                name=f'dense_{i+1}'
            ))

            # Batch Normalization
            if config['use_batch_norm']:
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))

            # Dropout
            if config['dropout_rate'] > 0:
                model.add(layers.Dropout(config['dropout_rate'], name=f'dropout_{i+1}'))

        # 输出层
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def create_cnn_model(self, input_dim: int, config: Dict[str, Any]) -> Model:
        """创建卷积神经网络模型（1D CNN）"""
        model = models.Sequential(name='CNN')

        # 重塑输入为1D卷积格式
        model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))

        # 卷积层
        for i, filters in enumerate(config['conv_filters']):
            model.add(layers.Conv1D(
                filters,
                kernel_size=config['kernel_size'],
                activation=config['activation'],
                padding='same',
                name=f'conv_{i+1}'
            ))

            if config['use_batch_norm']:
                model.add(layers.BatchNormalization(name=f'bn_conv_{i+1}'))

            model.add(layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}'))

            if config['dropout_rate'] > 0:
                model.add(layers.Dropout(config['dropout_rate'], name=f'dropout_conv_{i+1}'))

        # 展平
        model.add(layers.Flatten(name='flatten'))

        # 全连接层
        for i, units in enumerate(config['dense_layers']):
            model.add(layers.Dense(
                units,
                activation=config['activation'],
                kernel_regularizer=regularizers.l2(config['l2_reg']),
                name=f'dense_{i+1}'
            ))

            if config['dropout_rate'] > 0:
                model.add(layers.Dropout(config['dropout_rate'], name=f'dropout_dense_{i+1}'))

        # 输出层
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def create_lstm_model(self, input_dim: int, config: Dict[str, Any]) -> Model:
        """创建LSTM模型"""
        model = models.Sequential(name='LSTM')

        # 重塑输入
        model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))

        # LSTM层
        for i, units in enumerate(config['lstm_units']):
            return_sequences = i < len(config['lstm_units']) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=config['dropout_rate'],
                recurrent_dropout=config['recurrent_dropout'],
                name=f'lstm_{i+1}'
            ))

            if config['use_batch_norm']:
                model.add(layers.BatchNormalization(name=f'bn_lstm_{i+1}'))

        # 全连接层
        for i, units in enumerate(config['dense_layers']):
            model.add(layers.Dense(
                units,
                activation=config['activation'],
                kernel_regularizer=regularizers.l2(config['l2_reg']),
                name=f'dense_{i+1}'
            ))

            if config['dropout_rate'] > 0:
                model.add(layers.Dropout(config['dropout_rate'], name=f'dropout_{i+1}'))

        # 输出层
        model.add(layers.Dense(1, activation='sigmoid', name='output'))

        return model

    def get_optimizer(self, optimizer_name: str, learning_rate: float):
        """获取优化器"""
        optimizers = {
            'adam': Adam(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate, momentum=0.9),
            'rmsprop': RMSprop(learning_rate=learning_rate)
        }
        return optimizers.get(optimizer_name.lower(), Adam(learning_rate=learning_rate))

    def train_model(self, x_train, y_train, x_test, y_test,
                   model_type: str = 'dnn',
                   config: Dict[str, Any] = None):
        """训练深度学习模型"""
        try:
            if config is None:
                config = self.get_default_config(model_type)

            logging.info(f"====== 开始训练深度学习模型: {model_type.upper()} ======")
            logging.info(f"训练配置: {config}")

            # 获取输入维度
            input_dim = x_train.shape[1]

            # 创建模型
            if model_type.lower() == 'dnn':
                model = self.create_dnn_model(input_dim, config)
            elif model_type.lower() == 'cnn':
                model = self.create_cnn_model(input_dim, config)
            elif model_type.lower() == 'lstm':
                model = self.create_lstm_model(input_dim, config)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 编译模型
            optimizer = self.get_optimizer(config['optimizer'], config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy',
                        metrics.Precision(name='precision'),
                        metrics.Recall(name='recall'),
                        metrics.AUC(name='auc')]
            )

            # 打印模型结构
            model.summary(print_fn=lambda x: logging.info(x))

            # 回调函数
            callback_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

            # 训练模型
            logging.info("开始训练...")
            history = model.fit(
                x_train, y_train,
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                validation_data=(x_test, y_test),
                callbacks=callback_list,
                verbose=1
            )

            logging.info("训练完成！")

            # 评估模型
            y_train_pred = (model.predict(x_train) > 0.5).astype(int).flatten()
            train_metric = get_classification_metric(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = (model.predict(x_test) > 0.5).astype(int).flatten()
            test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"训练集指标: {train_metric}")
            logging.info(f"测试集指标: {test_metric}")

            # 如果有data_transformation_artifact，则保存模型
            if self.data_transformation_artifact is not None and self.model_trainer_config is not None:
                # 保存模型
                preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

                # 创建包装器以兼容现有的预测接口
                class KerasModelWrapper:
                    def __init__(self, keras_model, preprocessor):
                        self.model = keras_model
                        self.preprocessor = preprocessor

                    def predict(self, X):
                        X_transformed = self.preprocessor.transform(X)
                        predictions = (self.model.predict(X_transformed) > 0.5).astype(int).flatten()
                        return predictions

                wrapped_model = KerasModelWrapper(model, preprocessor)

                # 保存到artifact目录
                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=wrapped_model)

                # 保存到final_models目录
                final_model_dir = "final_models"
                os.makedirs(final_model_dir, exist_ok=True)

                # 保存Keras模型
                model.save(os.path.join(final_model_dir, f"dl_model_{model_type}.keras"))

                # 保存包装后的模型
                save_object(os.path.join(final_model_dir, "model.pkl"), wrapped_model)

                logging.info(f"====== 深度学习模型 {model_type.upper()} 训练完成 ======")

                return ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_metric_artifact=train_metric,
                    test_metric_artifact=test_metric
                )
            else:
                # 简化模式，仅返回模型和指标
                logging.info(f"====== 深度学习模型 {model_type.upper()} 训练完成 ======")
                return model, test_metric

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def get_default_config(self, model_type: str) -> Dict[str, Any]:
        """获取默认配置"""
        configs = {
            'dnn': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'l2_reg': 0.001,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10
            },
            'cnn': {
                'conv_filters': [64, 32],
                'kernel_size': 3,
                'dense_layers': [64, 32],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'l2_reg': 0.001,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10
            },
            'lstm': {
                'lstm_units': [64, 32],
                'dense_layers': [32],
                'activation': 'relu',
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.2,
                'use_batch_norm': True,
                'l2_reg': 0.001,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10
            }
        }
        return configs.get(model_type.lower(), configs['dnn'])

    def train(self, x_train, y_train, x_test, y_test):
        """
        简化的训练方法，用于独立测试

        参数:
            x_train: 训练特征
            y_train: 训练标签
            x_test: 测试特征
            y_test: 测试标签

        返回:
            model: 训练好的模型
            metrics: 测试集指标字典
        """
        return self.train_model(x_train, y_train, x_test, y_test,
                               self.model_type, self.config)

    def initiate_model_trainer(self, model_type: str = 'dnn',
                              config: Dict[str, Any] = None) -> ModelTrainerArtifact:
        """启动模型训练"""
        try:
            logging.info("开始加载转换后的数据...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("数据加载完毕，即将开始深度学习模型训练...")

            return self.train_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                model_type=model_type,
                config=config
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
