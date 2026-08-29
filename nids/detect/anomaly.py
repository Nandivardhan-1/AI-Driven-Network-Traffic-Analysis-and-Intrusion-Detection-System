"""Unsupervised anomaly layer.

The rule layer only fires on behaviour somebody already characterised, and the
supervised layer only recognises families it has labelled examples of. Both fail
on the same case: an attack nobody has written down yet. This layer is the answer
to that case.

An isolation forest is fitted **on benign training traffic only**. It never sees a
single attack, so it cannot have learned an attack signature -- it has learned only
what normal looks like, and it alerts on distance from normal. That makes it the
one component whose detections are not bounded by the label set, which is exactly
why the hybrid keeps it despite its weaker precision.

Two design choices worth stating because they affect how the output should be read:

* **The alert threshold is set from benign scores, not from attack scores.** It is
  the ``anomaly_alert_percentile`` quantile of the training-benign score
  distribution, so the expected benign alert rate is fixed by construction (1% at
  the 99th percentile) and is not tuned against the test set.
* **Confidence is a percentile, not a probability.** "More unusual than 99.4% of
  benign training traffic" is a statement the score can actually support; a
  calibrated probability of attack is not, because the model never saw attacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FUSION, RANDOM_SEED
from ..features import FeatureSpace
from ..ml.iforest import IsolationForest


class AnomalyDetector:
    """Isolation forest over the scaled feature space, fitted on benign rows."""

    def __init__(self, space: FeatureSpace, n_estimators: int = 150,
                 max_samples: int = 256, percentile: float | None = None,
                 seed: int = RANDOM_SEED) -> None:
        self.space = space
        self.forest = IsolationForest(n_estimators=n_estimators,
                                     max_samples=max_samples, random_state=seed)
        self.percentile = (FUSION["anomaly_alert_percentile"]
                           if percentile is None else percentile)
        self.threshold_ = 0.0
        self.benign_scores_: np.ndarray = np.zeros(0)
        self.benign_mean_: np.ndarray = np.zeros(0)
        self.benign_std_: np.ndarray = np.ones(1)
        self.n_benign_ = 0

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, benign_mask: np.ndarray | None = None
            ) -> "AnomalyDetector":
        """Fit on benign rows only.

        ``benign_mask`` defaults to ``df["category"] == "normal"``. Passing it
        explicitly is how the pcap path (which has no labels) supplies its own
        notion of a clean baseline period.
        """
        if benign_mask is None:
            if "category" not in df.columns:
                raise ValueError("no 'category' column; pass benign_mask explicitly")
            benign_mask = (df["category"].astype(str) == "normal").to_numpy()
        benign_mask = np.asarray(benign_mask, dtype=bool)
        if benign_mask.sum() < 50:
            raise ValueError(f"need >=50 benign rows to model normal, got {benign_mask.sum()}")

        X = self.space.transform_scaled(df)[benign_mask]
        self.n_benign_ = int(X.shape[0])
        self.forest.fit(X)
        self.benign_scores_ = np.sort(self.forest.score_samples(X))
        self.threshold_ = float(np.percentile(self.benign_scores_, self.percentile))
        # Raw (unscaled) benign statistics, so explanations can quote a feature's
        # observed value against the normal range in its own units.
        raw = self.space.transform(df)[benign_mask]
        self.benign_mean_ = raw.mean(axis=0)
        self.benign_std_ = raw.std(axis=0)
        self.benign_std_[self.benign_std_ < 1e-12] = 1.0
        return self

    # ------------------------------------------------------------------ #
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Anomaly score in (0, 1); higher is more isolated, hence more unusual."""
        return self.forest.score_samples(self.space.transform_scaled(df))

    def novelty(self, scores: np.ndarray) -> np.ndarray:
        """Fraction of training-benign traffic each score exceeds, in [0, 1].

        This is the number quoted to the analyst as confidence: it is a rank
        statistic against observed normal traffic, which the model can support.
        """
        if self.benign_scores_.size == 0:
            raise RuntimeError("AnomalyDetector must be fitted first")
        pos = np.searchsorted(self.benign_scores_, np.asarray(scores, float),
                              side="right")
        return pos / float(self.benign_scores_.size)

    def predict(self, df: pd.DataFrame) -> dict:
        scores = self.score(df)
        return {"score": scores,
                "novelty": self.novelty(scores),
                "alert": scores >= self.threshold_,
                "threshold": self.threshold_}

    # ------------------------------------------------------------------ #
    def explain(self, df: pd.DataFrame, rows: np.ndarray, top_k: int = 5) -> list[list[dict]]:
        """Per-row feature attribution for the isolation decision.

        Each entry names a feature, the value observed, the benign mean, and how
        many benign standard deviations away that is -- so "why is this unusual"
        is answered in the feature's own units rather than as an opaque score.
        """
        rows = np.asarray(rows, dtype=int)
        if rows.size == 0:
            return []
        sub = df.iloc[rows]
        Xs = self.space.transform_scaled(sub)
        Xr = self.space.transform(sub)
        weights = self.forest.isolation_attribution(Xs)
        names = self.space.feature_names
        out = []
        for r in range(rows.size):
            order = np.argsort(weights[r])[::-1][:top_k]
            items = []
            for j in order:
                z = (Xr[r, j] - self.benign_mean_[j]) / self.benign_std_[j]
                items.append({
                    "feature": names[j],
                    "group": self.space.group_of(names[j]),
                    "weight": float(weights[r, j]),
                    "observed": float(Xr[r, j]),
                    "benign_mean": float(self.benign_mean_[j]),
                    "benign_sd": float(self.benign_std_[j]),
                    "z": float(z),
                    "meaning": self.space.describe(names[j]),
                })
            out.append(items)
        return out
