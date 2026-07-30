"""
LUCID CNN model.
Based on doriguzzi/lucid-ddos (IEEE TNSM 2020).

1D CNN that learns spatiotemporal patterns from network flows
for DDoS detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class LucidCNN:
    """
    LUCID CNN DDoS detector.

    Architecture:
    - Input: (time_steps, n_features) flow samples.
    - Conv1D layers extract temporal features.
    - MaxPooling for dimensionality reduction.
    - Dense layers for classification.
    """
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'time_steps': 10,      # packets per time window
        'n_features': 11,      # features per packet
        'kernels': 64,         # convolution kernels
        'kernel_size': 3,      # kernel size
        'pool_size': 2,        # pooling size
        'dense_units': 64,     # dense layer units
        'dropout': 0.5,        # dropout rate
        'learning_rate': 0.001,
        'batch_size': 1024,
        'epochs': 100
    }
    
    def __init__(self, **kwargs):
        """Initialize LUCID CNN."""
        self.params = {**self.DEFAULT_PARAMS, **kwargs}
        self.model = None
        self.is_fitted = False
        self.history = None
        self._tf = None
    
    def _check_tensorflow(self):
        """Check whether TensorFlow is available."""
        if self._tf is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                self._tf = tf
            except ImportError:
                raise ImportError("LUCID requires TensorFlow. Install: pip install tensorflow")
        return self._tf
    
    def _build_model(self):
        """Build the CNN model."""
        tf = self._check_tensorflow()
        
        time_steps = self.params['time_steps']
        n_features = self.params['n_features']
        kernels = self.params['kernels']
        kernel_size = self.params['kernel_size']
        pool_size = self.params['pool_size']
        dense_units = self.params['dense_units']
        dropout = self.params['dropout']
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(time_steps, n_features)),
            
            # First conv block
            tf.keras.layers.Conv1D(kernels, kernel_size, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            
            # Second conv block
            tf.keras.layers.Conv1D(kernels * 2, kernel_size, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            
            # Flatten and dense
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dense_units // 2, activation='relu'),
            tf.keras.layers.Dropout(dropout / 2),
            
            # Output layer (binary: normal/DDoS)
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
        Train the model.
        
        Args:
            X: training data (n_samples, time_steps, n_features).
            y: labels (0=normal, 1=DDoS).
            X_val: validation data.
            y_val: validation labels.
            epochs: number of training epochs.
            batch_size: batch size.
            verbose: logging level.
        """
        tf = self._check_tensorflow()
        
        epochs = epochs or self.params['epochs']
        batch_size = batch_size or self.params['batch_size']
        
        # Update parameters
        if X.ndim == 2:
            # If 2D, reshape to 3D
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
        """Predict (returns 0 or 1)."""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.params['time_steps'], -1)
        
        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.params['time_steps'], -1)
        
        proba = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the model."""
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
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
        """Save the model."""
        if self.model:
            self.model.save(path)
    
    def load(self, path: str):
        """Load the model."""
        tf = self._check_tensorflow()
        self.model = tf.keras.models.load_model(path)
        self.is_fitted = True
