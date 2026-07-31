"""
LUCID detector — integrates CNN model and dataset parser.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time

from networksecurity.engine.lucid.cnn import LucidCNN
from networksecurity.engine.lucid.dataset_parser import LucidDatasetParser, FlowSample

logger = logging.getLogger(__name__)


@dataclass
class LucidResult:
    """LUCID detection result."""
    is_ddos: bool
    confidence: float
    flow_id: str = ""
    packets_analyzed: int = 0
    detection_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'is_ddos': self.is_ddos,
            'confidence': self.confidence,
            'flow_id': self.flow_id,
            'packets_analyzed': self.packets_analyzed,
            'detection_time_ms': self.detection_time_ms
        }


class LucidDetector:
    """
    LUCID DDoS detector.

    Detection pipeline:
    1. Dataset parsing: convert raw packets to flow samples.
    2. Feature extraction: extract temporal features.
    3. CNN classification: classify using trained CNN.

    Usage:
    ```python
    detector = LucidDetector()
    detector.train(X_train, y_train)
    
    for packet in packets:
        result = detector.process_packet(packet)
        if result and result.is_ddos:
            print(f"DDoS detected! confidence={result.confidence}")
    ```
    """
    
    def __init__(self, time_window: float = 10.0, packets_per_flow: int = 10, **cnn_params):
        """
        Args:
            time_window: time window in seconds.
            packets_per_flow: packets per flow sample.
            **cnn_params: CNN model parameters.
        """
        self.time_window = time_window
        self.packets_per_flow = packets_per_flow
        
        # Dataset parser
        self.parser = LucidDatasetParser(time_window, packets_per_flow)
        
        # CNN model
        cnn_params['time_steps'] = packets_per_flow
        cnn_params['n_features'] = self.parser.n_features
        self.cnn = LucidCNN(**cnn_params)
        
        # State
        self.is_trained = False
        self.total_packets = 0
        self.total_detections = 0
        self.ddos_detections = 0
    
    def set_attack_info(self, attackers: List[str], victims: List[str]):
        """Set attacker and victim IPs (for training labels)."""
        self.parser.set_attack_info(attackers, victims)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, verbose: int = 0) -> Dict:
        """
        Train the detector.
        
        Args:
            X: training data (n_samples, packets_per_flow, n_features).
            y: labels (0=normal, 1=DDoS).
        """
        result = self.cnn.fit(X, y, X_val, y_val, epochs=epochs, verbose=verbose)
        self.is_trained = True
        return result
    
    def train_from_packets(self, packets: List[Dict], epochs: int = 100, 
                           validation_split: float = 0.2, verbose: int = 0) -> Dict:
        """
        Train from raw packets.
        
        Args:
            packets: list of packet dicts.
            epochs: number of training epochs.
            validation_split: validation set ratio.
        """
        # Parse packets
        X, y = self.parser.parse_batch(packets)
        
        if len(X) == 0:
            raise ValueError("Not enough packets to generate training samples")
        
        # Split train/validation
        n_val = int(len(X) * validation_split)
        if n_val > 0:
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]
            X_train, X_val = X[n_val:], X[:n_val]
            y_train, y_val = y[n_val:], y[:n_val]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        return self.train(X_train, y_train, X_val, y_val, epochs, verbose)
    
    def process_packet(self, packet: Dict) -> Optional[LucidResult]:
        """
        Process a single packet.

        Returns:
            Detection result if the flow is complete, else None.
        """
        self.total_packets += 1
        
        result = self.parser.process_packet(packet)
        if result is None:
            return None
        
        sample, _ = result
        return self._detect(sample)
    
    def _detect(self, sample: np.ndarray) -> LucidResult:
        """Run detection."""
        start_time = time.time()
        self.total_detections += 1
        
        if not self.is_trained:
            if not getattr(self, "_warned_untrained", False):
                logger.warning(
                    "LUCID detector called but model is not trained. "
                    "Run detector.train() or detector.load() first."
                )
                self._warned_untrained = True
            return LucidResult(
                is_ddos=False,
                confidence=0.0,
                packets_analyzed=self.packets_per_flow,
                detection_time_ms=0.0,
            )
        
        # Predict
        sample_batch = sample.reshape(1, self.packets_per_flow, -1)
        proba = self.cnn.predict_proba(sample_batch)[0]
        is_ddos = proba[1] > 0.5
        
        if is_ddos:
            self.ddos_detections += 1
        
        detection_time = (time.time() - start_time) * 1000
        
        return LucidResult(
            is_ddos=is_ddos,
            confidence=float(proba[1]),
            packets_analyzed=self.packets_per_flow,
            detection_time_ms=detection_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction (sklearn-compatible)."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.cnn.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.cnn.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the model."""
        return self.cnn.evaluate(X, y)
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'total_packets': self.total_packets,
            'total_detections': self.total_detections,
            'ddos_detections': self.ddos_detections,
            'ddos_rate': self.ddos_detections / max(1, self.total_detections),
            'is_trained': self.is_trained
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_packets = 0
        self.total_detections = 0
        self.ddos_detections = 0
    
    def save(self, path: str):
        """Save the model."""
        self.cnn.save(path)
    
    def load(self, path: str):
        """Load the model."""
        self.cnn.load(path)
        self.is_trained = True
