"""Standard network traffic dataset loader.

Supports common IDS dataset formats:
- NSL-KDD (CSV with header)
- CICIDS2017 (CSV with header)
- UNSW-NB15 (CSV with header)
"""

from __future__ import annotations

import logging
from pathlib import Path

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

# Class labels that denote BENIGN / normal traffic across supported datasets.
# Used to build the attack (1) vs benign (0) mask regardless of dataset naming.
_BENIGN_LABELS: set[str] = {"normal", "benign"}


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

    @staticmethod
    def _encode_labels(y_raw: pd.Series) -> np.ndarray:
        """Encode a target column into binary attack(1)/benign(0) labels.

        Semantics (shared by ``load`` and ``train_test_split`` so the two
        public entry points always agree):
        - An explicit benign string ("normal", "normal.", "BENIGN", ...) -> 0.
        - A numeric column (or a column of numeric strings) -> 0 means benign,
          anything else is an attack.  This matches UNSW-NB15's 0/1 convention
          and also handles a CSV that loads numeric labels as ``object`` dtype.
        - Mixed columns (benign keyword present *and* numeric values present):
          a benign keyword -> 0 AND a numeric 0 -> 0; everything else -> attack.
          This keeps the documented "numeric 0 is always benign" guarantee even
          when a benign keyword also appears in the column, while still treating
          the benign keyword itself as benign in mixed data.
        - Anything else (no benign label and not purely numeric) -> attack.
        """
        # Numeric view of the column (coerces non-numeric to NaN).
        y_num = pd.to_numeric(y_raw, errors="coerce")
        # String view, lowercased, trimmed (drops a trailing '.' like "normal.").
        y_str = y_raw.astype(str).str.strip().str.rstrip(".")
        benign_str = y_str.str.lower().isin(_BENIGN_LABELS)

        if benign_str.any():
            # A recognizable benign label is present: take string semantics, but
            # also honor numeric 0 as benign so mixed columns behave sanely.
            if y_raw.isna().any():
                logger.warning("Dataset contains NaN labels; treated as attack.")
            benign = benign_str
            if y_num.notna().any():
                benign = benign | (y_num == 0)
            # Treat the NaN sentinel (used by train_test_split) as attack, not benign.
            benign = benign & (y_str != "__nan_attack__")
            return (~benign).astype(int).values

        if y_num.notna().any():
            if y_num.isna().any():
                logger.warning(
                    "Target column has NaN/missing labels; those rows are "
                    "treated as attack (1)."
                )
                y_num = y_num.fillna(1)
            return (y_num != 0).astype(int).values

        logger.warning(
            "Target column has no recognized benign label and is not numeric; "
            "all samples labeled as attack."
        )
        return np.ones(len(y_raw), dtype=int)

    def load(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load a dataset, returning (X, y)."""
        path = Path(path)
        # Parquet is the distribution format of the bundled datasets
        # (datasets/unsw-nb15/*.parquet); CSV is read by default otherwise.
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
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

        y = self._encode_labels(y_raw)

        # Handle non-numeric features.
        # NOTE: callers needing aligned train/test encodings (e.g. a model
        # trained on one split and evaluated on another) must use
        # train_test_split below, which fits get_dummies on the training split
        # and applies the same columns to the test split (aligning dimensions).
        if X.shape[1] > 0:
            X = pd.get_dummies(X, drop_first=True)
        # CICIDS2017 is known to carry Infinity in Flow Duration/Packets rates;
        # fillna(0) does not touch inf, which would crash sklearn/Keras downstream.
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        return X.values, y

    def _encode(self, train: pd.DataFrame, test: pd.DataFrame, target: str | None = None):
        """Fit one-hot encoding on train, transform test with aligned columns."""
        if target is None:
            target = self._config.get("target_column", "label")
        if target not in train.columns or target not in test.columns:
            raise ValueError(
                f"Target column '{target}' missing. "
                f"train cols: {list(train.columns)}, test cols: {list(test.columns)}"
            )

        y_train = self._encode_labels(train[target])
        y_test = self._encode_labels(test[target])

        # Drop metadata columns (same set as load()) so features do not leak
        # label-related columns such as UNSW 'attack_cat' or NSL-KDD 'difficulty'.
        for col in self._config.get("drop_columns", []):
            if col in train.columns:
                train = train.drop(columns=[col])
            if col in test.columns:
                test = test.drop(columns=[col])

        X_train = train.drop(columns=[target])
        X_test = test.drop(columns=[target])

        if X_train.shape[1] > 0:
            X_train = pd.get_dummies(X_train, drop_first=True)
        if X_test.shape[1] > 0:
            X_test = pd.get_dummies(X_test, drop_first=True)
        # Align test columns to the training set (union, fill missing with 0).
        # If train encoded to zero columns (no features left), fall back to an
        # empty zero-column frame so reindex does not fail.
        if X_train.shape[1] == 0:
            X_train = pd.DataFrame(index=X_train.index)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        return X_train.values, y_train, X_test.values, y_test

    def train_test_split(
        self, path: str | Path, test_size: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split as tts

        path = Path(path)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
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

        # sklearn cannot stratify on NaN and would raise (or treat NaN as its own
        # stratum), so we fill NaN labels in-place with a sentinel that
        # _encode_labels maps to attack(1). This keeps the rows (consistent with
        # load(), which also keeps and encodes NaN as attack) instead of dropping
        # them, while still allowing safe stratification on the remaining classes.
        if df[target].isna().any():
            logger.warning(
                "Target column '%s' has NaN/missing labels; those rows are "
                "treated as attack (1).", target
            )
            df = df.copy()
            df[target] = df[target].fillna("__nan_attack__")

        stratify = None
        counts = df[target].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            # Guard against sklearn raising when a class is too small to be split
            # at the requested test_size (e.g. a 2-member class with test_size>=0.5).
            min_test = int(np.ceil(test_size * counts.min()))
            min_train = counts.min() - min_test
            if min_test >= 1 and min_train >= 1:
                stratify = df[target]

        train_df, test_df = tts(
            df, test_size=test_size, random_state=42, stratify=stratify
        )
        return self._encode(train_df, test_df, target=target)
