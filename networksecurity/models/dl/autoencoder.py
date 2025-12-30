"""
AutoEncoder异常检测模型
用于无监督异常检测
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import pandas as pd
import logging
import time

from networksecurity.models.dl.base import DLModelBase, DLModelRegistry, DLModelInfo, DLModelType, DLTrainingResult

logger = logging.getLogger(__name__)


@DLModelRegistry.register("autoencoder")
class AutoEncoderDetector(DLModelBase):
    """AutoEncoder异常检测器"""
    
    def __init__(self, input_dim: int = None, threshold_percentile: float = 95, **kwargs):
        super().__init__(input_dim, num_classes=2, **kwargs)
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="AutoEncoder",
            type=DLModelType.AUTOENCODER,
            description="自编码器异常检测，通过重构误差检测异常",
            supports_proba=False,
            default_params={'encoding_dim': 16, 'hidden_layers': [64, 32]}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        encoding_dim = self.params.get('encoding_dim', 16)
        hidden_layers = self.params.get('hidden_layers', [64, 32])
        
        # Encoder
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        x = inputs
        for units in hidden_layers:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(x)
        
        # Decoder
        x = encoded
        for units in reversed(hidden_layers):
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        decoded = tf.keras.layers.Dense(self.input_dim, activation='linear')(x)
        
        model = tf.keras.Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X, y=None, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0) -> DLTrainingResult:
        """训练AutoEncoder（无监督）"""
        start_time = time.time()
        try:
            tf = self._check_tensorflow()
            X = np.array(X) if not isinstance(X, np.ndarray) else X
            
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            
            self.model = self._build_model()
            
            validation_data = (X_val, X_val) if X_val is not None else None
            if validation_data:
                X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
                validation_data = (X_val, X_val)
            
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            
            self.history = self.model.fit(X, X, epochs=epochs, batch_size=batch_size,
                                          validation_data=validation_data, callbacks=callbacks, verbose=verbose)
            
            # 计算阈值
            reconstructions = self.model.predict(X, verbose=0)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            self.threshold = np.percentile(mse, self.threshold_percentile)
            
            self.is_fitted = True
            self._training_time = time.time() - start_time
            
            return DLTrainingResult(
                success=True, model_name=self.info.name, train_time=self._training_time,
                final_loss=float(self.history.history['loss'][-1])
            )
        except Exception as e:
            return DLTrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X) -> np.ndarray:
        """返回1(正常)或-1(异常)"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return np.where(mse > self.threshold, -1, 1)
    
    def reconstruction_error(self, X) -> np.ndarray:
        """返回重构误差"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.power(X - reconstructions, 2), axis=1)


@DLModelRegistry.register("vae")
class VAEDetector(DLModelBase):
    """变分自编码器异常检测器"""
    
    def __init__(self, input_dim: int = None, threshold_percentile: float = 95, **kwargs):
        super().__init__(input_dim, num_classes=2, **kwargs)
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.encoder = None
        self.decoder = None
    
    @property
    def info(self) -> DLModelInfo:
        return DLModelInfo(
            name="VAE",
            type=DLModelType.VAE,
            description="变分自编码器，生成式异常检测",
            supports_proba=False,
            default_params={'latent_dim': 8, 'hidden_dim': 32}
        )
    
    def _build_model(self):
        tf = self._check_tensorflow()
        latent_dim = self.params.get('latent_dim', 8)
        hidden_dim = self.params.get('hidden_dim', 32)
        
        # Encoder
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(hidden_dim, activation='relu')(inputs)
        z_mean = tf.keras.layers.Dense(latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(latent_dim)(x)
        
        # Sampling
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
        
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_hidden = tf.keras.layers.Dense(hidden_dim, activation='relu')
        decoder_output = tf.keras.layers.Dense(self.input_dim)
        
        x = decoder_hidden(z)
        outputs = decoder_output(x)
        
        # VAE Model
        vae = tf.keras.Model(inputs, outputs)
        
        # Loss
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(inputs - outputs))
        kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var))
        vae.add_loss(reconstruction_loss + 0.001 * kl_loss)
        vae.compile(optimizer='adam')
        
        return vae
    
    def fit(self, X, y=None, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0) -> DLTrainingResult:
        start_time = time.time()
        try:
            tf = self._check_tensorflow()
            X = np.array(X) if not isinstance(X, np.ndarray) else X
            
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            
            self.model = self._build_model()
            
            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            self.history = self.model.fit(X, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
            
            reconstructions = self.model.predict(X, verbose=0)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            self.threshold = np.percentile(mse, self.threshold_percentile)
            
            self.is_fitted = True
            self._training_time = time.time() - start_time
            
            return DLTrainingResult(success=True, model_name=self.info.name, train_time=self._training_time)
        except Exception as e:
            return DLTrainingResult(success=False, model_name=self.info.name, error=str(e))
    
    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型未训练")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return np.where(mse > self.threshold, -1, 1)
