"""
KitNET — Autoencoder ensemble for anomaly detection.
Based on ymirsky/Kitsune-py (NDSS'18).

Uses an ensemble of small autoencoders, each learning normal
patterns for a subset of features.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AutoEncoder:
    """
    Lightweight autoencoder with a single hidden layer
    for fast training and inference.
    """
    
    def __init__(self, input_dim: int, hidden_ratio: float = 0.75, 
                 learning_rate: float = 0.1):
        """
        Args:
            input_dim: Input dimension.
            hidden_ratio: Hidden-layer size ratio relative to input.
            learning_rate: Learning rate.
        """
        self.input_dim = input_dim
        self.hidden_dim = max(1, int(input_dim * hidden_ratio))
        self.learning_rate = learning_rate
        
        # Initialize weights (Xavier initialization).
        limit = np.sqrt(6.0 / (input_dim + self.hidden_dim))
        self.W_encode = np.random.uniform(-limit, limit, (input_dim, self.hidden_dim))
        self.b_encode = np.zeros(self.hidden_dim)
        
        limit = np.sqrt(6.0 / (self.hidden_dim + input_dim))
        self.W_decode = np.random.uniform(-limit, limit, (self.hidden_dim, input_dim))
        self.b_decode = np.zeros(input_dim)
        
        # Normalization parameters.
        self.norm_mean = None
        self.norm_std = None
        self.is_fitted = False
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative."""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input."""
        if self.norm_mean is None:
            return x
        return (x - self.norm_mean) / (self.norm_std + 1e-10)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode."""
        return self._sigmoid(np.dot(x, self.W_encode) + self.b_encode)
    
    def decode(self, h: np.ndarray) -> np.ndarray:
        """Decode."""
        return self._sigmoid(np.dot(h, self.W_decode) + self.b_decode)
    
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass. Returns (reconstruction, hidden, normalized_input)."""
        x_norm = self._normalize(x)
        h = self.encode(x_norm)
        x_recon = self.decode(h)
        return x_recon, h, x_norm
    
    def train_step(self, x: np.ndarray) -> float:
        """Single training step. Returns reconstruction error."""
        x_recon, h, x_norm = self.forward(x)
        
        # Compute error.
        error = x_norm - x_recon
        rmse = np.sqrt(np.mean(error ** 2))
        
        # Backpropagation.
        d_decode = error * self._sigmoid_derivative(
            np.dot(h, self.W_decode) + self.b_decode
        )
        d_encode = np.dot(d_decode, self.W_decode.T) * self._sigmoid_derivative(
            np.dot(x_norm, self.W_encode) + self.b_encode
        )
        
        # Update weights.
        self.W_decode += self.learning_rate * np.outer(h, d_decode)
        self.b_decode += self.learning_rate * d_decode
        self.W_encode += self.learning_rate * np.outer(x_norm, d_encode)
        self.b_encode += self.learning_rate * d_encode
        
        return rmse
    
    def compute_rmse(self, x: np.ndarray) -> float:
        """Compute reconstruction RMSE."""
        x_recon, _, x_norm = self.forward(x)
        return np.sqrt(np.mean((x_norm - x_recon) ** 2))
    
    def fit_normalization(self, X: np.ndarray):
        """Fit normalization parameters."""
        self.norm_mean = np.mean(X, axis=0)
        self.norm_std = np.std(X, axis=0)
        self.norm_std[self.norm_std < 1e-10] = 1.0


class KitNET:
    """
    KitNET — Autoencoder ensemble.

    Architecture:
    1. Feature mapping: cluster input features into small groups.
    2. Ensemble: one autoencoder per feature group, running in parallel.
    3. Output: one autoencoder aggregating all ensemble RMSE outputs.
    """

    def __init__(self, input_dim: int, max_autoencoder_size: int = 10,
                 fm_grace_period: int = 5000, ad_grace_period: int = 50000,
                 learning_rate: float = 0.1, hidden_ratio: float = 0.75,
                 threshold_percentile: float = 99.0):
        """
        Args:
            input_dim: Input feature dimension.
            max_autoencoder_size: Max input dimension per autoencoder.
            fm_grace_period: Feature mapping learning period (packets).
            ad_grace_period: Anomaly detection training period (packets).
            learning_rate: Learning rate.
            hidden_ratio: Hidden layer ratio.
        """
        self.input_dim = input_dim
        self.max_ae_size = max_autoencoder_size
        self.fm_grace = fm_grace_period
        self.ad_grace = ad_grace_period
        self.learning_rate = learning_rate
        self.hidden_ratio = hidden_ratio
        self.threshold_percentile = threshold_percentile

        # Feature map
        self.feature_map: list[list[int]] = []

        # Ensemble autoencoders
        self.ensemble: list[AutoEncoder] = []

        # Output autoencoder
        self.output_ae: AutoEncoder | None = None

        # Training state
        self.n_trained = 0
        self.fm_data: list[np.ndarray] = []
        self.is_fm_done = False
        self.is_ad_done = False

        # Anomaly threshold
        self.threshold = None
        self.rmse_history: list[float] = []
    
    def _build_feature_map(self, X: np.ndarray):
        """Build feature map via correlation clustering."""
        n_features = X.shape[1]
        
        # Compute feature correlation matrix.
        corr_matrix = np.corrcoef(X.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Greedy clustering.
        assigned = set()
        self.feature_map = []
        
        for i in range(n_features):
            if i in assigned:
                continue
            
            # Create new group.
            group = [i]
            assigned.add(i)
            
            # Find correlated features.
            correlations = np.abs(corr_matrix[i])
            sorted_indices = np.argsort(correlations)[::-1]
            
            for j in sorted_indices:
                if j in assigned:
                    continue
                if len(group) >= self.max_ae_size:
                    break
                if correlations[j] > 0.3:  # correlation threshold.
                    group.append(j)
                    assigned.add(j)
            
            self.feature_map.append(group)
        
        logger.info(f"KitNET: Created {len(self.feature_map)} feature groups")
    
    def _build_ensemble(self):
        """Build the autoencoder ensemble."""
        self.ensemble = []
        for group in self.feature_map:
            ae = AutoEncoder(
                input_dim=len(group),
                hidden_ratio=self.hidden_ratio,
                learning_rate=self.learning_rate,
            )
            self.ensemble.append(ae)

        # Output autoencoder
        self.output_ae = AutoEncoder(
            input_dim=len(self.ensemble),
            hidden_ratio=self.hidden_ratio,
            learning_rate=self.learning_rate,
        )

        logger.info("KitNET: Built %d ensemble autoencoders", len(self.ensemble))
    
    def process(self, x: np.ndarray) -> float:
        """Process a single sample.

        Returns 0 during training; returns anomaly score (RMSE) after training.
        """
        self.n_trained += 1

        # Feature mapping learning period
        if self.n_trained <= self.fm_grace:
            self.fm_data.append(x.copy())
            return 0.0

        # Complete feature mapping
        if not self.is_fm_done:
            fm_array = np.array(self.fm_data)
            self._build_feature_map(fm_array)
            self._build_ensemble()

            # Fit normalization for each autoencoder
            for i, group in enumerate(self.feature_map):
                group_data = fm_array[:, group]
                self.ensemble[i].fit_normalization(group_data)

            # Fit normalization for the output autoencoder on the ensemble
            # RMSE vectors recorded during the FM grace period.  Without this
            # the output layer's inputs are never normalized, hurting its
            # convergence (see _train_step, which feeds raw ensemble RMSEs).
            ensemble_fm_rmses = np.array([
                [self.ensemble[i].compute_rmse(fm_array[j, group])
                 for i, group in enumerate(self.feature_map)]
                for j in range(len(fm_array))
            ], dtype=np.float32)
            self.output_ae.fit_normalization(ensemble_fm_rmses)

            self.fm_data = []  # Free memory
            self.is_fm_done = True

        # Anomaly detection training period
        if self.n_trained <= self.fm_grace + self.ad_grace:
            rmse = self._train_step(x)
            self.rmse_history.append(rmse)
            return 0.0

        # Training complete — set threshold
        if not self.is_ad_done:
            if self.rmse_history:
                self.threshold = np.percentile(
                    self.rmse_history, self.threshold_percentile
                )
            else:
                self.threshold = 1.0
            self.rmse_history = []
            self.is_ad_done = True
            logger.info("KitNET: training complete, threshold=%.4f", self.threshold)

        # Run detection
        return self._execute(x)
    
    def _train_step(self, x: np.ndarray) -> float:
        """Training step."""
        ensemble_rmses = []
        
        for i, group in enumerate(self.feature_map):
            x_group = x[group]
            rmse = self.ensemble[i].train_step(x_group)
            ensemble_rmses.append(rmse)
        
        # Train output layer.
        ensemble_rmses = np.array(ensemble_rmses)
        output_rmse = self.output_ae.train_step(ensemble_rmses)
        
        return output_rmse
    
    def _execute(self, x: np.ndarray) -> float:
        """Run detection and return anomaly score."""
        ensemble_rmses = []
        
        for i, group in enumerate(self.feature_map):
            x_group = x[group]
            rmse = self.ensemble[i].compute_rmse(x_group)
            ensemble_rmses.append(rmse)
        
        ensemble_rmses = np.array(ensemble_rmses)
        output_rmse = self.output_ae.compute_rmse(ensemble_rmses)
        
        return output_rmse
    
    def is_anomaly(self, rmse: float) -> bool:
        """Check whether the sample is anomalous."""
        if self.threshold is None:
            return False
        return rmse > self.threshold
    
    def get_state(self) -> dict:
        """Get model state."""
        return {
            'n_trained': self.n_trained,
            'is_fm_done': self.is_fm_done,
            'is_ad_done': self.is_ad_done,
            'threshold': self.threshold,
            'n_ensembles': len(self.ensemble) if self.ensemble else 0
        }
