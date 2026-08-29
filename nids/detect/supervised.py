"""Supervised classification layer.

Where the rule layer answers "does this match something we wrote down" and the
anomaly layer answers "is this unlike normal", this layer answers the question an
analyst actually asks first: **which of the four attack classes is this, and how
sure are you.**

It is a random forest over the 55-column feature space, trained on KDDTrain+ with
balanced class weights. Two properties earn it its place in the hybrid:

* It generalises within a family. A rule for ``neptune`` fires on the exact
  threshold it was given; the forest learns the region of feature space around it,
  so a variant that shifts one feature is still classified.
* Its predictions decompose exactly. ``contributions()`` returns a per-feature
  additive breakdown that sums, with the bias, to the predicted probability, so
  every alert carries a ranked list of the features that produced it rather than a
  bare class name.

Its weakness is equally structural and is reported rather than hidden: it can only
recognise classes it was trained on, it is weakest on the classes with fewest
training examples (R2L 0.79% and U2R 0.04% of KDDTrain+), and it is the component
most exposed to the train/test distribution shift NSL-KDD deliberately contains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CATEGORIES, FUSION, RANDOM_SEED
from ..features import FeatureSpace
from ..ml.forest import RandomForestClassifier
from ..ml.preprocess import LabelEncoder


class SupervisedClassifier:
    """Random-forest multiclass detector with additive per-alert explanations."""

    def __init__(self, space: FeatureSpace, n_estimators: int = 120,
                 max_depth: int = 20, min_samples_leaf: int = 2,
                 threshold: float | None = None, seed: int = RANDOM_SEED) -> None:
        self.space = space
        self.forest = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features="sqrt",
            class_weight="balanced", random_state=seed)
        self.labels = LabelEncoder()
        self.threshold = (FUSION["supervised_alert_threshold"]
                          if threshold is None else threshold)
        self.classes_: list[str] = []
        self.normal_index_ = 0

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, target: str = "category") -> "SupervisedClassifier":
        if target not in df.columns:
            raise ValueError(f"training frame has no '{target}' column")
        y_text = df[target].astype(str).to_numpy(dtype=object)
        y = self.labels.fit(y_text).transform(y_text)
        X = self.space.transform(df)
        self.forest.fit(X, y)
        self.classes_ = list(self.labels.classes_)
        self.normal_index_ = (self.classes_.index("normal")
                              if "normal" in self.classes_ else -1)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> dict:
        """Class probabilities plus the derived attack probability and decision."""
        proba = self.forest.predict_proba(self.space.transform(df))
        best = proba.argmax(axis=1)
        category = np.array([self.classes_[i] for i in best], dtype=object)
        if self.normal_index_ >= 0:
            attack_prob = 1.0 - proba[:, self.normal_index_]
        else:
            attack_prob = proba.max(axis=1)
        return {
            "proba": proba,
            "classes": list(self.classes_),
            "category": category,
            "confidence": proba.max(axis=1),
            "attack_prob": attack_prob,
            "alert": attack_prob >= self.threshold,
            "threshold": self.threshold,
        }

    @property
    def oob_score_(self) -> float | None:
        return self.forest.oob_score_

    def importances(self) -> pd.DataFrame:
        """Global feature importances, annotated with group and justification."""
        imp = self.forest.feature_importances_
        rows = [{"feature": name,
                 "group": self.space.group_of(name),
                 "importance": float(imp[i]),
                 "meaning": self.space.describe(name)}
                for i, name in enumerate(self.space.feature_names)]
        return (pd.DataFrame(rows).sort_values("importance", ascending=False)
                .reset_index(drop=True))

    # ------------------------------------------------------------------ #
    def explain(self, df: pd.DataFrame, rows: np.ndarray, top_k: int = 5,
                target_class: np.ndarray | None = None) -> list[list[dict]]:
        """Per-row additive attribution toward the predicted (or given) class.

        ``contributions`` is O(rows x trees x depth) in Python, so this is called
        on alerted rows only -- never on the full test set.
        """
        rows = np.asarray(rows, dtype=int)
        if rows.size == 0:
            return []
        sub = df.iloc[rows]
        X = self.space.transform(sub)
        bias, contrib = self.forest.contributions(X)
        proba = bias + contrib.sum(axis=1)
        if target_class is None:
            k_idx = proba.argmax(axis=1)
        else:
            k_idx = np.array([self.classes_.index(str(c)) for c in target_class])
        names = self.space.feature_names
        out = []
        for r in range(rows.size):
            k = int(k_idx[r])
            col = contrib[r, :, k]
            order = np.argsort(np.abs(col))[::-1][:top_k]
            items = []
            for j in order:
                items.append({
                    "feature": names[j],
                    "group": self.space.group_of(names[j]),
                    "contribution": float(col[j]),
                    "observed": float(X[r, j]),
                    "direction": "raises" if col[j] > 0 else "lowers",
                    "meaning": self.space.describe(names[j]),
                })
            out.append({
                "predicted": self.classes_[k],
                "probability": float(proba[r, k]),
                "baseline": float(bias[k]),
                "features": items,
            })
        return out

    def check_additivity(self, df: pd.DataFrame, rows: np.ndarray) -> float:
        """Max absolute gap between reconstructed and direct probabilities.

        A correctness assertion for the explanation path: if this is not ~1e-15 the
        per-feature story does not actually add up to the decision being explained.
        """
        sub = df.iloc[np.asarray(rows, dtype=int)]
        X = self.space.transform(sub)
        bias, contrib = self.forest.contributions(X)
        return float(np.abs((bias + contrib.sum(axis=1))
                            - self.forest.predict_proba(X)).max())


CATEGORY_ORDER = list(CATEGORIES)
