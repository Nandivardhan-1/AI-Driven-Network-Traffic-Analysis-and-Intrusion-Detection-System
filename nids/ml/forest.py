"""Random forest classifier with per-prediction feature attribution.

Two things beyond a stock forest matter for this project:

* ``feature_importances_`` -- global impurity-decrease ranking, used to justify
  the feature set in the report.
* ``contributions`` -- per-prediction attribution computed by walking each tree's
  decision path and accumulating the change in class probability at every split
  (the "tree interpreter" decomposition). Summed over trees this gives an exact
  additive breakdown ``P(attack) = bias + sum_j contribution_j``, which is what
  the alert explanations in Section D are built from.
"""
from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier, HistBinner


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 18,
        min_samples_leaf: int = 2,
        max_features: int | str | None = "sqrt",
        max_bins: int = 64,
        bootstrap: bool = True,
        class_weight: str | None = "balanced",
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        self.binner: HistBinner | None = None
        self.trees_: list[DecisionTreeClassifier] = []
        self.n_classes_ = 0
        self.oob_score_: float | None = None

    # ------------------------------------------------------------------ #
    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        if self.class_weight != "balanced":
            return np.ones(len(y))
        with np.errstate(divide="ignore"):
            w = len(y) / (self.n_classes_ * np.where(counts > 0, counts, 1))
        return w[y]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        self.n_classes_ = int(y.max()) + 1
        self.binner = HistBinner(max_bins=self.max_bins).fit(X)
        Xb = self.binner.transform(X)
        base_w = self._class_weights(y)
        rng = np.random.default_rng(self.random_state)

        oob_votes = np.zeros((n, self.n_classes_))
        oob_seen = np.zeros(n, dtype=bool)
        self.trees_ = []
        for t in range(self.n_estimators):
            tree_rng = np.random.default_rng(self.random_state + 1000 + t)
            if self.bootstrap:
                boot = tree_rng.integers(0, n, size=n)
                counts = np.bincount(boot, minlength=n).astype(float)
                w = base_w * counts
                in_bag = counts > 0
            else:
                w = base_w
                in_bag = np.ones(n, dtype=bool)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_bins=self.binner.n_bins,
                rng=tree_rng,
            )
            tree.fit(Xb, y, self.n_classes_, sample_weight=w)
            self.trees_.append(tree)
            if self.bootstrap and (~in_bag).any():
                oob_idx = np.flatnonzero(~in_bag)
                oob_votes[oob_idx] += tree.predict_proba(Xb[oob_idx])
                oob_seen[oob_idx] = True
        if oob_seen.any():
            pred = oob_votes[oob_seen].argmax(axis=1)
            self.oob_score_ = float((pred == y[oob_seen]).mean())
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self.binner.transform(np.asarray(X, dtype=np.float64))
        acc = np.zeros((Xb.shape[0], self.n_classes_))
        for tree in self.trees_:
            acc += tree.predict_proba(Xb)
        return acc / len(self.trees_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean impurity decrease per feature, normalised to sum to 1."""
        n_features = len(self.binner.edges_)
        imp = np.zeros(n_features)
        for tree in self.trees_:
            gini = 1.0 - (tree.value ** 2).sum(axis=1)
            internal = np.flatnonzero(tree.feature >= 0)
            for node in internal:
                l, r = tree.left[node], tree.right[node]
                n_p, n_l, n_r = (tree.n_node_samples[node],
                                 tree.n_node_samples[l], tree.n_node_samples[r])
                dec = n_p * gini[node] - n_l * gini[l] - n_r * gini[r]
                imp[tree.feature[node]] += max(dec, 0.0)
        total = imp.sum()
        return imp / total if total > 0 else imp

    def contributions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Additive per-feature attribution.

        Returns ``(bias, contrib)`` where ``bias`` has shape (n_classes,) and
        ``contrib`` has shape (n_rows, n_features, n_classes). By construction
        ``predict_proba(X) == bias + contrib.sum(axis=1)`` up to float error.
        """
        Xb = self.binner.transform(np.asarray(X, dtype=np.float64))
        n_rows, n_features = Xb.shape
        contrib = np.zeros((n_rows, n_features, self.n_classes_))
        bias = np.zeros(self.n_classes_)
        for tree in self.trees_:
            bias += tree.value[0]
            for i in range(n_rows):
                node = 0
                row = Xb[i]
                while tree.feature[node] >= 0:
                    f = int(tree.feature[node])
                    nxt = int(tree.left[node] if row[f] <= tree.threshold_bin[node]
                              else tree.right[node])
                    contrib[i, f] += tree.value[nxt] - tree.value[node]
                    node = nxt
        return bias / len(self.trees_), contrib / len(self.trees_)
