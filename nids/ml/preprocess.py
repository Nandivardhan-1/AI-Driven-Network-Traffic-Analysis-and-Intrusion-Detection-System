"""Preprocessing primitives implemented on NumPy only.

The prototype deliberately avoids scikit-learn so that the whole detection path
-- including the learning algorithms -- is inspectable and runs with nothing but
numpy/pandas installed. Each transformer follows the familiar
``fit``/``transform`` contract so the code reads like standard practice.
"""
from __future__ import annotations

import numpy as np


class StandardScaler:
    """Zero-mean, unit-variance scaling with safe handling of constant columns."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OrdinalEncoder:
    """Map string categories to integer codes, with a reserved code for unseen.

    Unseen categories at inference time are mapped to ``len(categories_)`` rather
    than raising. On NSL-KDD this matters: KDDTest+ contains services that never
    appear in KDDTrain+, and silently dropping those rows would flatter the
    evaluation.
    """

    def __init__(self) -> None:
        self.categories_: list[str] = []
        self._lookup: dict[str, int] = {}

    def fit(self, values) -> "OrdinalEncoder":
        uniq = sorted({str(v) for v in np.asarray(values).ravel()})
        self.categories_ = uniq
        self._lookup = {c: i for i, c in enumerate(uniq)}
        return self

    @property
    def unseen_code(self) -> int:
        return len(self.categories_)

    def transform(self, values) -> np.ndarray:
        unseen = self.unseen_code
        return np.array(
            [self._lookup.get(str(v), unseen) for v in np.asarray(values).ravel()],
            dtype=np.int64,
        )

    def fit_transform(self, values) -> np.ndarray:
        return self.fit(values).transform(values)

    def inverse(self, code: int) -> str:
        if 0 <= code < len(self.categories_):
            return self.categories_[code]
        return "<unseen>"


class LabelEncoder(OrdinalEncoder):
    """Alias with intent: used for the target vector rather than a feature."""

    @property
    def classes_(self) -> list[str]:
        return self.categories_


def stratified_split(y: np.ndarray, val_fraction: float = 0.2, seed: int = 0):
    """Split preserving class proportions -- important for the rare U2R class."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * (1.0 - val_fraction))))
        train_parts.append(idx[:cut])
        val_parts.append(idx[cut:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate([p for p in val_parts if len(p)])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx
