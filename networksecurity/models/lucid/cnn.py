"""
LUCID CNN模型
基于 doriguzzi/lucid-ddos 实现

使用CNN学习网络流量的时空特征，检测DDoS攻击。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class LucidCNN:
    """
    LUCID CNN DDoS检测器
    
    架构:
    - 输入: (time_steps, n_features) 的流量样本
    - Conv1D层提取时序特征
    - MaxPooling降维
    - Dense层分类
    """
    
    # 默认超参数
    DEFAULT_PARAMS = {
        'time_steps': 10,      # 时间窗口内的数据包数
        'n_features': 11,      # 每个数据包的特征数
        'kernels': 64,         # 卷积核数量
        'kernel_size': 3,      # 卷积核大小
        'pool_size': 2,        # 池化大小
        'dense_units': 64,     # 全连接层单元数
        'dropout': 0.5,        # Dropout率
        'learning_rate': 0.001,
        'batch_size': 1024,
        'epochs': 100
    }
    
    def __init__(self, **kwargs):
        """初始化LUCID CNN"""
        self.params = {**self.DEFAULT_PARAMS, **kwargs}
        self.model = None
        self.is_fitted = False
        self.history = None
        self._tf = None
    
    def _check_tensorflow(self):
        """检查TensorFlow是否可用"""
        if self._tf is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                self._tf = tf
            except ImportError:
                raise ImportError("LUCID需要TensorFlow，请安装: pip install tensorflow")
        return self._tf
    
    def _build_model(self):
        """构建CNN模型"""
        tf = self._check_tensorflow()
        
        time_steps = self.params['time_steps']
        n_features = self.params['n_features']
        kernels = self.params['kernels']
        kernel_size = self.params['kernel_size']
        pool_size = self.params['pool_size']
        dense_units = self.params['dense_units']
        dropout = self.params['dropout']
        
        model = tf.keras.Sequential([
            # 输入层
            tf.keras.layers.Input(shape=(time_steps, n_features)),
            
            # 第一个卷积块
            tf.keras.layers.Conv1D(kernels, kernel_size, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            
            # 第二个卷积块
            tf.keras.layers.Conv1D(kernels * 2, kernel_size, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            
            # 展平和全连接
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dense_units // 2, activation='relu'),
            tf.keras.layers.Dropout(dropout / 2),
            
            # 输出层 (二分类: 正常/DDoS)
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = None, batch_size: int = None, verbose: int = 0) -> Dict:
        """
        训练模型
        
        Args:
            X: 训练数据 (n_samples, time_steps, n_features)
            y: 标签 (0=正常, 1=DDoS)
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 日志级别
        """
        tf = self._check_tensorflow()
        
        epochs = epochs or self.params['epochs']
        batch_size = batch_size or self.params['batch_size']
        
        # 更新参数
        if X.ndim == 2:
            # 如果是2D，reshape为3D
            X = X.reshape(X.shape[0], self.params['time_steps'], -1)
            self.params['n_features'] = X.shape[2]
        else:
            self.params['time_steps'] = X.shape[1]
            self.params['n_features'] = X.shape[2]
        
        self.model = self._build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        return {
            'loss': self.history.history['loss'][-1],
            'accuracy': self.history.history['accuracy'][-1],
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 (返回0或1)"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.params['time_steps'], -1)
        
        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.params['time_steps'], -1)
        
        proba = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.params['time_steps'], -1)
        
        results = self.model.evaluate(X, y, verbose=0)
        return {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
    
    def save(self, path: str):
        """保存模型"""
        if self.model:
            self.model.save(path)
    
    def load(self, path: str):
        """加载模型"""
        tf = self._check_tensorflow()
        self.model = tf.keras.models.load_model(path)
        self.is_fitted = True
