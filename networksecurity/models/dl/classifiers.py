"""
DL分类器实现
包含DNN、CNN、LSTM等深度学习分类模型
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from networksecurity.models.dl.base import DLModelBase, DLModelRegistry, DLModelInfo, DLModelType

logger = logging.getLogger(__name__)


@DLModelRegistry.register("dnn")
class DNNClassifier(DLModelBase):
    """深度神经网络分类器"""
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="DNN",
            type=DLModelType.DNN,
            description="多层全连接神经网络",
            default_params={'hidden_layers': [128, 64, 32], 'dropout': 0.3}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        hidden_layers = self.params.get('hidden_layers', [128, 64, 32])
        dropout = self.params.get('dropout', 0.3)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))
        
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout))
        
        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model


@DLModelRegistry.register("cnn1d")
class CNN1DClassifier(DLModelBase):
    """1D卷积神经网络分类器"""
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="CNN-1D",
            type=DLModelType.CNN,
            description="一维卷积神经网络，适合序列特征",
            default_params={'filters': [64, 128], 'kernel_size': 3}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        filters = self.params.get('filters', [64, 128])
        kernel_size = self.params.get('kernel_size', 3)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim, 1)))
        
        for f in filters:
            model.add(tf.keras.layers.Conv1D(f, kernel_size, activation='relu', padding='same'))
            model.add(tf.keras.layers.MaxPooling1D(2))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        
        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        if X_val is not None:
            X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        return super().fit(X, y, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return super().predict(X)


@DLModelRegistry.register("lstm")
class LSTMClassifier(DLModelBase):
    """LSTM分类器"""
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="LSTM",
            type=DLModelType.LSTM,
            description="长短期记忆网络，适合时序数据",
            default_params={'units': [64, 32], 'dropout': 0.3}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        units = self.params.get('units', [64, 32])
        dropout = self.params.get('dropout', 0.3)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim, 1)))
        
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            model.add(tf.keras.layers.LSTM(u, return_sequences=return_seq))
            model.add(tf.keras.layers.Dropout(dropout))
        
        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        if X_val is not None:
            X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        return super().fit(X, y, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return super().predict(X)


@DLModelRegistry.register("bilstm")
class BiLSTMClassifier(DLModelBase):
    """双向LSTM分类器"""
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="BiLSTM",
            type=DLModelType.LSTM,
            description="双向长短期记忆网络",
            default_params={'units': 64, 'dropout': 0.3}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        units = self.params.get('units', 64)
        dropout = self.params.get('dropout', 0.3)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim, 1)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units // 2)),
            tf.keras.layers.Dropout(dropout),
        ])
        
        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        if X_val is not None:
            X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        return super().fit(X, y, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return super().predict(X)


@DLModelRegistry.register("gru")
class GRUClassifier(DLModelBase):
    """GRU分类器"""
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="GRU",
            type=DLModelType.GRU,
            description="门控循环单元网络",
            default_params={'units': [64, 32], 'dropout': 0.3}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        units = self.params.get('units', [64, 32])
        dropout = self.params.get('dropout', 0.3)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_dim, 1)))
        
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            model.add(tf.keras.layers.GRU(u, return_sequences=return_seq))
            model.add(tf.keras.layers.Dropout(dropout))
        
        if self.num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        if X_val is not None:
            X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        return super().fit(X, y, X_val, y_val, **kwargs)
    
    def predict(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return super().predict(X)
