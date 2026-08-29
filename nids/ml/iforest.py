"""Isolation Forest anomaly detector (NumPy only).

Trained on benign traffic alone, so it needs no attack labels and can in
principle flag behaviour that no signature and no supervised model has seen. The
score is the standard ``2 ** (-E[h(x)] / c(psi))``: short average isolation depth
means easy to isolate, means anomalous.
"""
from __future__ import annotations

import numpy as np


def _c(n: int) -> float:
    """Average path length of an unsuccessful BST search over n points."""
    if n <= 1:
        return 1e-9
    if n == 2:
        return 1.0
    h = np.log(n - 1) + 0.5772156649015329
    return float(2.0 * h - 2.0 * (n - 1) / n)


class _ITree:
    __slots__ = ("feature", "threshold", "left", "right", "size", "depth_limit")

    def __init__(self, depth_limit: int) -> None:
        self.feature: list[int] = []
        self.threshold: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.size: list[int] = []
        self.depth_limit = depth_limit

    def _new_node(self, feature, threshold, size):
        self.feature.append(feature)
        self.threshold.append(threshold)
        self.left.append(-1)
        self.right.append(-1)
        self.size.append(size)
        return len(self.feature) - 1

    def build(self, X: np.ndarray, rng: np.random.Generator) -> "_ITree":
        stack = [(np.arange(X.shape[0]), 0, -1, False)]
        while stack:
            idx, depth, parent, is_left = stack.pop()
            n = len(idx)
            if depth >= self.depth_limit or n <= 1:
                node = self._new_node(-1, np.nan, n)
            else:
                sub = X[idx]
                lo, hi = sub.min(axis=0), sub.max(axis=0)
                spread = np.flatnonzero(hi > lo)
                if spread.size == 0:
                    node = self._new_node(-1, np.nan, n)
                else:
                    f = int(rng.choice(spread))
                    thr = float(rng.uniform(lo[f], hi[f]))
                    node = self._new_node(f, thr, n)
                    mask = sub[:, f] < thr
                    stack.append((idx[~mask], depth + 1, node, False))
                    stack.append((idx[mask], depth + 1, node, True))
            if parent >= 0:
                if is_left:
                    self.left[parent] = node
                else:
                    self.right[parent] = node
        self.feature = np.asarray(self.feature, dtype=np.int64)
        self.threshold = np.asarray(self.threshold, dtype=np.float64)
        self.left = np.asarray(self.left, dtype=np.int64)
        self.right = np.asarray(self.right, dtype=np.int64)
        self.size = np.asarray(self.size, dtype=np.int64)
        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        depth = np.zeros(n, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        while active.any():
            rows = np.flatnonzero(active)
            cur = node[rows]
            internal = self.feature[cur] >= 0
            rows = rows[internal]
            if rows.size == 0:
                break
            cur = node[rows]
            go_left = X[rows, self.feature[cur]] < self.threshold[cur]
            node[rows] = np.where(go_left, self.left[cur], self.right[cur])
            depth[rows] += 1.0
            active = np.zeros(n, dtype=bool)
            active[rows] = True
        return depth + np.array([_c(s) for s in self.size[node]])


class IsolationForest:
    """Unsupervised anomaly scorer fitted on benign traffic only."""

    def __init__(self, n_estimators: int = 120, max_samples: int = 256,
                 random_state: int = 0) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees_: list[_ITree] = []
        self._c_psi = 1.0
        self.n_features_ = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        self.n_features_ = X.shape[1]
        psi = int(min(self.max_samples, n))
        self._c_psi = _c(psi)
        depth_limit = max(1, int(np.ceil(np.log2(max(psi, 2)))))
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            sub = X[rng.choice(n, size=psi, replace=False)] if psi < n else X
            self.trees_.append(_ITree(depth_limit).build(sub, rng))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in (0, 1); higher means more anomalous."""
        X = np.asarray(X, dtype=np.float64)
        total = np.zeros(X.shape[0])
        for tree in self.trees_:
            total += tree.path_length(X)
        mean_depth = total / len(self.trees_)
        return 2.0 ** (-mean_depth / self._c_psi)

    def isolation_attribution(self, X: np.ndarray) -> np.ndarray:
        """Which features isolated each row, as a (n_rows, n_features) weight.

        Random feature selection means simply counting splits per feature would
        mostly measure the sampler, not the data. Instead a split credits its
        feature in proportion to how much it *shrank* the surviving population,
        discounted by depth: ``(1 - n_child / n_parent) / (depth + 1)``. A feature
        that drops a row into a nearly empty branch near the root scores high; a
        feature that sends it down the majority branch scores ~0.

        Cost is O(rows x trees x depth) in Python, so call it on alerted rows only.
        """
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((X.shape[0], self.n_features_))
        for tree in self.trees_:
            for i in range(X.shape[0]):
                node, depth = 0, 0
                while tree.feature[node] >= 0:
                    f = int(tree.feature[node])
                    nxt = int(tree.left[node] if X[i, f] < tree.threshold[node]
                              else tree.right[node])
                    parent_n = max(int(tree.size[node]), 1)
                    shrink = 1.0 - int(tree.size[nxt]) / parent_n
                    out[i, f] += shrink / (depth + 1.0)
                    node, depth = nxt, depth + 1
        rows = out.sum(axis=1, keepdims=True)
        return out / np.where(rows > 0, rows, 1.0)


