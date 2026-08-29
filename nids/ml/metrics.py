"""Evaluation metrics (NumPy only).

Binary metrics are reported from the analyst's point of view: the positive class
is "malicious", so recall is the fraction of attacks caught and 1 - precision is
the share of alerts that waste an analyst's time.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def binary_scores(y_true, y_pred) -> dict:
    """TP/FP/FN/TN plus precision, recall, F1, FPR and accuracy."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "support_positive": tp + fn,
        "support_negative": tn + fp,
    }


def per_class_report(y_true, y_pred, class_names: list[str]) -> list[dict]:
    """One-vs-rest precision/recall/F1/support for every class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    rows = []
    for i, name in enumerate(class_names):
        s = binary_scores(y_true == i, y_pred == i)
        rows.append({
            "class": name,
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "support": int(np.sum(y_true == i)),
            "detected": s["tp"],
        })
    return rows


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    return float(np.mean([
        binary_scores(np.asarray(y_true) == i, np.asarray(y_pred) == i)["f1"]
        for i in range(n_classes)
    ]))


def roc_curve(y_true, scores):
    """Return (fpr, tpr, thresholds) without any external dependency."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(~y)
    p, n = max(int(y.sum()), 1), max(int((~y).sum()), 1)
    tpr = np.concatenate([[0.0], tps / p, [1.0]])
    fpr = np.concatenate([[0.0], fps / n, [1.0]])
    thr = np.concatenate([[np.inf], scores[order], [-np.inf]])
    return fpr, tpr, thr


def auc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def roc_auc(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def precision_recall_curve(y_true, scores):
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(~y)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(int(y.sum()), 1)
    return precision, recall, scores[order]


def format_confusion(cm: np.ndarray, class_names: list[str]) -> str:
    """Fixed-width confusion matrix for terminal output."""
    w = max(9, max(len(c) for c in class_names) + 1)
    head = " " * (w + 2) + "".join(f"{c:>{w}}" for c in class_names)
    lines = [f"{'':>{w}}  predicted ->", head]
    for i, name in enumerate(class_names):
        lines.append(f"{name:>{w}}  " + "".join(f"{v:>{w}}" for v in cm[i]))
    return "\n".join(lines)

