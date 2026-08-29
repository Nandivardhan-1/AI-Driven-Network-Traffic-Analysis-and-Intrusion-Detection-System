"""Histogram-based CART decision tree (NumPy only).

Continuous features are pre-binned into at most ``max_bins`` quantile buckets
once, up front. Split search then reduces to accumulating a
``(n_bins, n_classes)`` histogram with ``np.bincount`` and scanning its prefix
sums, which is what makes a from-scratch forest fast enough to train on the full
125k-row NSL-KDD training set in seconds rather than minutes.
"""
from __future__ import annotations

import numpy as np


class HistBinner:
    """Quantile binning of a float matrix into uint8 bin codes."""

    def __init__(self, max_bins: int = 64) -> None:
        if not 2 <= max_bins <= 255:
            raise ValueError("max_bins must be in [2, 255]")
        self.max_bins = max_bins
        self.edges_: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "HistBinner":
        X = np.asarray(X, dtype=np.float64)
        self.edges_ = []
        qs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            col = X[:, j]
            edges = np.unique(np.quantile(col, qs))
            # Degenerate/near-constant columns collapse to a single bin.
            self.edges_.append(edges.astype(np.float64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.edges_):
            out[:, j] = np.searchsorted(edges, X[:, j], side="left")
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def threshold_value(self, feature: int, bin_index: int) -> float:
        """Real-valued split point for a bin index, for human-readable rules."""
        edges = self.edges_[feature]
        if len(edges) == 0:
            return float("nan")
        return float(edges[min(bin_index, len(edges) - 1)])

    @property
    def n_bins(self) -> int:
        return self.max_bins + 1


class DecisionTreeClassifier:
    """CART classifier grown on pre-binned features, Gini criterion.

    Stored as flat parallel arrays so that prediction and path-walking are cheap
    and the structure can be serialised with ``numpy.savez``.
    """

    def __init__(
        self,
        max_depth: int = 16,
        min_samples_leaf: int = 2,
        min_samples_split: int = 4,
        max_features: int | str | None = "sqrt",
        n_bins: int = 65,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_bins = n_bins
        self.rng = rng or np.random.default_rng(0)
        # Flat node arrays, filled during fit.
        self.feature: list[int] = []
        self.threshold_bin: list[int] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.value: list[np.ndarray] = []
        self.n_node_samples: list[float] = []
        self.n_classes_ = 0

    # ------------------------------------------------------------------ #
    def _n_features_to_try(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return max(1, min(int(mf), n_features))

    def _best_split(self, Xb, y, w, idx, features):
        """Return (feature, threshold_bin, score) maximising the Gini surrogate.

        Minimising weighted child Gini is equivalent to maximising
        ``sum(L^2)/|L| + sum(R^2)/|R|`` over class-count vectors L and R, which is
        what we scan here.
        """
        K, B = self.n_classes_, self.n_bins
        yi = y[idx]
        wi = w[idx]
        best = (-1, -1, -np.inf)
        for f in features:
            codes = Xb[idx, f].astype(np.int64) * K + yi
            hist = np.bincount(codes, weights=wi, minlength=B * K).reshape(B, K)
            cum = np.cumsum(hist, axis=0)
            total = cum[-1]
            nl = cum.sum(axis=1)
            nr = total.sum() - nl
            ok = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not ok.any():
                continue
            right = total - cum
            with np.errstate(divide="ignore", invalid="ignore"):
                score = np.where(
                    ok,
                    (cum ** 2).sum(axis=1) / np.where(nl > 0, nl, 1)
                    + (right ** 2).sum(axis=1) / np.where(nr > 0, nr, 1),
                    -np.inf,
                )
            b = int(np.argmax(score))
            if score[b] > best[2]:
                best = (int(f), b, float(score[b]))
        return best

    def _add_leaf(self, counts: np.ndarray) -> int:
        total = counts.sum()
        probs = counts / total if total > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)
        self.feature.append(-1)
        self.threshold_bin.append(-1)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(probs)
        self.n_node_samples.append(float(total))
        return len(self.feature) - 1

    # ------------------------------------------------------------------ #
    def fit(self, Xb: np.ndarray, y: np.ndarray, n_classes: int,
            sample_weight: np.ndarray | None = None) -> "DecisionTreeClassifier":
        Xb = np.asarray(Xb, dtype=np.uint8)
        y = np.asarray(y, dtype=np.int64)
        n, n_features = Xb.shape
        self.n_classes_ = int(n_classes)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        k_try = self._n_features_to_try(n_features)

        root_counts = np.bincount(y, weights=w, minlength=n_classes)
        # Stack entries: (sample_idx, depth, parent_node, is_left, counts)
        stack = [(np.arange(n), 0, -1, False, root_counts)]
        while stack:
            idx, depth, parent, is_left, counts = stack.pop()
            total = counts.sum()
            pure = (counts > 0).sum() <= 1
            node = None
            if (depth >= self.max_depth or pure or len(idx) < self.min_samples_split
                    or len(idx) < 2 * self.min_samples_leaf):
                node = self._add_leaf(counts)
            else:
                feats = self.rng.choice(n_features, size=k_try, replace=False)
                f, b, score = self._best_split(Xb, y, w, idx, feats)
                parent_score = float((counts ** 2).sum() / total) if total else 0.0
                if f < 0 or score <= parent_score + 1e-9:
                    node = self._add_leaf(counts)
                else:
                    probs = counts / total
                    self.feature.append(f)
                    self.threshold_bin.append(b)
                    self.left.append(-1)
                    self.right.append(-1)
                    self.value.append(probs)
                    self.n_node_samples.append(float(total))
                    node = len(self.feature) - 1
                    mask = Xb[idx, f] <= b
                    li, ri = idx[mask], idx[~mask]
                    lc = np.bincount(y[li], weights=w[li], minlength=n_classes)
                    rc = np.bincount(y[ri], weights=w[ri], minlength=n_classes)
                    stack.append((ri, depth + 1, node, False, rc))
                    stack.append((li, depth + 1, node, True, lc))
            if parent >= 0:
                if is_left:
                    self.left[parent] = node
                else:
                    self.right[parent] = node
        self._finalise()
        return self

    def _finalise(self) -> None:
        self.feature = np.asarray(self.feature, dtype=np.int64)
        self.threshold_bin = np.asarray(self.threshold_bin, dtype=np.int64)
        self.left = np.asarray(self.left, dtype=np.int64)
        self.right = np.asarray(self.right, dtype=np.int64)
        self.value = np.vstack(self.value) if len(self.value) else np.zeros((0, 0))
        self.n_node_samples = np.asarray(self.n_node_samples, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def apply(self, Xb: np.ndarray) -> np.ndarray:
        """Vectorised descent: return the leaf node index for every row."""
        Xb = np.asarray(Xb, dtype=np.uint8)
        node = np.zeros(Xb.shape[0], dtype=np.int64)
        active = np.ones(Xb.shape[0], dtype=bool)
        while active.any():
            cur = node[active]
            internal = self.feature[cur] >= 0
            if not internal.any():
                break
            rows = np.flatnonzero(active)[internal]
            cur = node[rows]
            go_left = Xb[rows, self.feature[cur]] <= self.threshold_bin[cur]
            node[rows] = np.where(go_left, self.left[cur], self.right[cur])
            still = np.zeros(Xb.shape[0], dtype=bool)
            still[rows] = True
            active = still
        return node

    def predict_proba(self, Xb: np.ndarray) -> np.ndarray:
        return self.value[self.apply(Xb)]

    def path(self, xb_row: np.ndarray) -> list[tuple[int, int, bool]]:
        """Decision path for one row as [(node, feature, went_left), ...]."""
        out, node = [], 0
        while self.feature[node] >= 0:
            f = int(self.feature[node])
            went_left = bool(xb_row[f] <= self.threshold_bin[node])
            out.append((node, f, went_left))
            node = int(self.left[node] if went_left else self.right[node])
        return out
