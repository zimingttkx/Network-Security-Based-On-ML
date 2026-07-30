"""Standard network traffic dataset loader.

Supports common IDS dataset formats:
- NSL-KDD (CSV with header)
- CICIDS2017 (CSV with header)
- UNSW-NB15 (CSV with header)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column mappings for common datasets
DATASET_CONFIGS: dict[str, dict] = {
    "nsl-kdd": {
        "target_column": "label",
        "drop_columns": ["difficulty"],
    },
    "unsw-nb15": {
        "target_column": "label",
        "drop_columns": ["id", "attack_cat"],
    },
}


class DatasetLoader:
    """Load and preprocess standard IDS datasets."""

    def __init__(self, dataset_type: str = "nsl-kdd") -> None:
        self._type = dataset_type
        self._config = DATASET_CONFIGS.get(dataset_type, {})

    def load(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a dataset, returning (X, y)."""
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows, %d columns", path, len(df), len(df.columns))

        # Drop metadata columns
        for col in self._config.get("drop_columns", []):
            if col in df.columns:
                df = df.drop(columns=[col])

        target = self._config.get("target_column", "label")
        if target not in df.columns:
            # Try common target column names
            for candidate in ["label", "Label", "class", "Class", "attack", "Attack"]:
                if candidate in df.columns:
                    target = candidate
                    break

        if target not in df.columns:
            raise ValueError(
                f"Target column not found. Available: {list(df.columns)}"
            )

        y_raw = df[target]
        X = df.drop(columns=[target])

        # Encode labels
        if y_raw.dtype == object:
            y = (y_raw != "normal").astype(int).values
        else:
            y = (y_raw != 0).astype(int).values

        # Handle non-numeric features
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(0).astype(np.float32)

        return X.values, y

    def train_test_split(
        self, path: str | Path, test_size: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split as tts

        X, y = self.load(path)
        return tts(X, y, test_size=test_size, random_state=42, stratify=y)
