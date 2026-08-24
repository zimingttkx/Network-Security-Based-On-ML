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
    "cicids2017": {
        "target_column": "Label",
        # CICIDS2017 also carries a flow-ID / timestamp header we never use
        "drop_columns": ["Flow ID", "Timestamp", "Source IP", "Destination IP"],
    },
}


class DatasetLoader:
    """Load and preprocess standard IDS datasets."""

    def __init__(self, dataset_type: str = "nsl-kdd") -> None:
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset_type '{dataset_type}'. "
                f"Supported: {list(DATASET_CONFIGS)}"
            )
        self._type = dataset_type
        self._config = DATASET_CONFIGS[dataset_type]

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

        # Encode labels. NSL-KDD test set uses "normal." (trailing dot) while
        # the train set uses "normal"; strip surrounding whitespace and any
        # trailing dot so both are treated as the same "normal" class.
        y_norm = y_raw.astype(str).str.strip().str.rstrip(".")
        if y_raw.dtype == object or y_norm.str.lower().isin(["normal"]).any():
            y = (y_norm.str.lower() != "normal").astype(int).values
        else:
            y = (y_raw != 0).astype(int).values

        # Handle non-numeric features.
        # NOTE: callers needing aligned train/test encodings (e.g. a model
        # trained on one split and evaluated on another) must use
        # train_test_split below, which fits get_dummies on the training split
        # and applies the same columns to the test split (aligning dimensions).
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(0).astype(np.float32)

        return X.values, y

    def _encode(self, train: pd.DataFrame, test: pd.DataFrame):
        """Fit one-hot encoding on train, transform test with aligned columns."""
        target = self._config.get("target_column", "label")
        if target not in train.columns or target not in test.columns:
            raise ValueError(
                f"Target column '{target}' missing. "
                f"train cols: {list(train.columns)}, test cols: {list(test.columns)}"
            )

        y_train = (train[target].astype(str).str.strip().str.rstrip(".")
                   .str.lower() != "normal").astype(int).values
        y_test = (test[target].astype(str).str.strip().str.rstrip(".")
                  .str.lower() != "normal").astype(int).values

        X_train = train.drop(columns=[target])
        X_test = test.drop(columns=[target])

        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)
        # Align test columns to the training set (union, fill missing with 0).
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_train = X_train.fillna(0).astype(np.float32)
        X_test = X_test.fillna(0).astype(np.float32)
        return X_train.values, y_train, X_test.values, y_test

    def train_test_split(
        self, path: str | Path, test_size: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split as tts

        df = pd.read_csv(path)
        target = self._config.get("target_column", "label")
        if target not in df.columns:
            for candidate in ["label", "Label", "class", "Class", "attack", "Attack"]:
                if candidate in df.columns:
                    target = candidate
                    break
        if target not in df.columns:
            raise ValueError(
                f"Target column not found. Available: {list(df.columns)}"
            )

        stratify = None
        counts = df[target].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = df[target]

        train_df, test_df = tts(
            df, test_size=test_size, random_state=42, stratify=stratify
        )
        return self._encode(train_df, test_df)
