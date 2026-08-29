"""Evaluation harness: everything section 4 of the brief asks for, computed once.

Design decisions that make the numbers meaningful rather than flattering:

* **Layer ablation is reported, not just the hybrid.** Each layer is scored alone on
  the same test split, so the claim "the hybrid is worth the complexity" is
  measured against its own components instead of asserted.
* **Recall is broken out per attack family, and families absent from training are
  marked.** A single recall figure hides the only interesting failure mode -- that
  detection collapses on families the models never saw.
* **Concrete TP / FP / FN records are extracted, with full explanations**, because
  aggregate metrics cannot show whether an alert would have been actionable.
* **Throughput is measured on the same run**, since section 5 asks about the cost
  of inline detection and a rows-per-second figure from the actual pipeline is the
  only honest basis for that discussion.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, CATEGORIES, NOVEL_TEST_ATTACKS
from .detect.hybrid import HybridDetector
from .ml.metrics import (accuracy, binary_scores, confusion_matrix, format_confusion,
                         macro_f1, per_class_report, precision_recall_curve, roc_curve,
                         roc_auc)


def _binary(truth: np.ndarray, pred: np.ndarray) -> dict:
    s = binary_scores(truth.astype(int), pred.astype(int))
    return {k: (round(float(v), 6) if isinstance(v, float) else int(v))
            for k, v in s.items()}


def _precision_at_k(truth: np.ndarray, order: np.ndarray,
                    ks=(50, 100, 250, 500, 1000, 2000, 5000)) -> list[dict]:
    out = []
    for k in ks:
        if k > order.size:
            continue
        sel = order[:k]
        out.append({"k": int(k),
                    "precision": round(float(truth[sel].mean()), 4),
                    "true_positives": int(truth[sel].sum())})
    return out


def _family_table(df: pd.DataFrame, result, truth: np.ndarray,
                  novel: set[str]) -> list[dict]:
    """Per-family recall for the hybrid and for each layer independently."""
    labels = df["label"].astype(str).to_numpy(dtype=object)
    cats = df["category"].astype(str).to_numpy(dtype=object)
    rule_fired = result.rule_score > 0
    rows = []
    for fam in sorted(set(labels[truth])):
        m = labels == fam
        rows.append({
            "family": fam,
            "category": str(cats[m][0]),
            "support": int(m.sum()),
            "in_training": fam not in novel,
            "hybrid_recall": round(float(result.is_alert[m].mean()), 4),
            "signature_recall": round(float(rule_fired[m].mean()), 4),
            "supervised_recall": round(float(result.supervised_alert[m].mean()), 4),
            "anomaly_recall": round(float(result.anomaly_alert[m].mean()), 4),
        })
    return sorted(rows, key=lambda r: (r["category"], -r["support"]))


def _severity_table(result, truth: np.ndarray) -> list[dict]:
    rows = []
    for sev in ("critical", "high", "medium", "low"):
        m = (result.severity == sev) & result.is_alert
        if not m.any():
            continue
        rows.append({"severity": sev, "alerts": int(m.sum()),
                     "precision": round(float(truth[m].mean()), 4),
                     "share_of_queue": round(float(m.sum() / result.is_alert.sum()), 4)})
    return rows


def _examples(det: HybridDetector, df: pd.DataFrame, result, truth: np.ndarray,
              n: int = 3) -> dict:
    """Concrete true-positive, false-positive and false-negative records."""
    order = result.alert_rows
    tp = np.array([i for i in order.tolist() if truth[i]][:n])
    fp = np.array([i for i in order.tolist() if not truth[i]][:n])
    missed = np.flatnonzero(truth & ~result.is_alert)
    # Rank misses by how close they came, so the examples are the informative
    # near-misses rather than arbitrary rows.
    fn = missed[np.argsort(-result.anomaly_novelty[missed])][:n] if missed.size else missed
    out = {}
    for name, idx in (("true_positive", tp), ("false_positive", fp),
                      ("false_negative", fn)):
        if idx.size == 0:
            out[name] = []
            continue
        if name == "false_negative":
            recs = det.explain_alerts(df, result, rows=idx, limit=None)
        else:
            recs = det.explain_alerts(df, result, rows=idx, limit=None)
        out[name] = recs
    return out


def evaluate(det: HybridDetector, train: pd.DataFrame, test: pd.DataFrame,
             provenance: dict, n_examples: int = 3) -> dict:
    """Run the full evaluation and return a JSON-serialisable report dict."""
    t0 = time.perf_counter()
    result = det.predict(test)
    wall = time.perf_counter() - t0

    truth = (test["category"].astype(str) != "normal").to_numpy()
    novel = set(provenance.get("novel_in_test", NOVEL_TEST_ATTACKS))

    # ---------------- layer ablation on the identical split ---------------- #
    ablation = {
        "signature_only": _binary(truth, result.rule_score > 0),
        "supervised_only": _binary(truth, result.supervised_alert),
        "anomaly_only": _binary(truth, result.anomaly_alert),
        "hybrid": _binary(truth, result.is_alert),
    }
    # Pairwise unions, to show where the hybrid's recall actually comes from.
    ablation["signature_or_supervised"] = _binary(
        truth, (result.rule_score > 0) | result.supervised_alert)
    ablation["signature_or_anomaly"] = _binary(
        truth, (result.rule_score > 0) | result.anomaly_alert)

    # ---------------- multiclass view (supervised layer) ------------------- #
    classes = list(result.classes)
    y_true = np.array([classes.index(c) if c in classes else -1
                       for c in test["category"].astype(str)])
    y_pred = np.array([classes.index(c) for c in result.supervised_category])
    keep = y_true >= 0
    cm = confusion_matrix(y_true[keep], y_pred[keep], len(classes))
    multiclass = {
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "confusion_text": format_confusion(cm, classes),
        "accuracy": round(float(accuracy(y_true[keep], y_pred[keep])), 4),
        "macro_f1": round(float(macro_f1(y_true[keep], y_pred[keep], len(classes))), 4),
        "per_class": [{k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()}
                      for r in per_class_report(y_true[keep], y_pred[keep], classes)],
    }

    # ---------------- ranking quality and score curves --------------------- #
    fpr_a, tpr_a, _ = roc_curve(truth.astype(int), result.anomaly_score)
    fpr_s, tpr_s, _ = roc_curve(truth.astype(int), result.supervised_attack_prob)
    prec_h, rec_h, _ = precision_recall_curve(truth.astype(int), result.fused_score)

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "source": provenance.get("source"),
            "name": provenance.get("name"),
            "note": provenance.get("note"),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_categories": {k: int(v) for k, v in
                                 train["category"].value_counts().items()},
            "test_categories": {k: int(v) for k, v in
                                test["category"].value_counts().items()},
            "families_absent_from_training": sorted(novel),
            "is_simulated": provenance.get("source") == "simulated",
        },
        "fit": det.fit_info,
        "ablation": ablation,
        "multiclass": multiclass,
        "families": _family_table(test, result, truth, novel),
        "severity": _severity_table(result, truth),
        "precision_at_k": _precision_at_k(truth, result.alert_rows),
        "score_auc": {
            "anomaly_roc_auc": round(float(roc_auc(truth.astype(int),
                                                   result.anomaly_score)), 4),
            "supervised_roc_auc": round(float(roc_auc(truth.astype(int),
                                              result.supervised_attack_prob)), 4),
            "hybrid_roc_auc": round(float(roc_auc(truth.astype(int),
                                                  result.fused_score)), 4),
        },
        "curves": {
            "anomaly_roc": {"fpr": fpr_a.tolist(), "tpr": tpr_a.tolist()},
            "supervised_roc": {"fpr": fpr_s.tolist(), "tpr": tpr_s.tolist()},
            "hybrid_pr": {"precision": prec_h.tolist(), "recall": rec_h.tolist()},
        },
        "rule_coverage": det.signatures.coverage(test).round(4)
                            .to_dict(orient="records"),
        "queue": result.summary(),
        "performance": {
            "test_rows": int(len(test)),
            "wall_seconds": round(wall, 3),
            "rows_per_second": round(len(test) / max(1e-9, wall), 1),
            "per_stage_seconds": {k: round(float(v), 4)
                                  for k, v in result.timing.items()},
            "explanation_cost_note": (
                "Per-alert explanations use exact tree traversal in Python and are "
                "computed on alerted rows only; the throughput above excludes them."),
        },
        "examples": _examples(det, test, result, truth, n=n_examples),
        "importances": det.supervised.importances().head(20).round(5)
                          .to_dict(orient="records"),
    }
    return report


def save_report(report: dict, path: str | None = None) -> str:
    path = path or os.path.join(ARTIFACT_DIR, "evaluation.json")

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_default)
    return path


def queue_sample(result, truth: np.ndarray | None = None, top: int = 200,
                 per_severity: int = 40, false_positives: int = 60,
                 seed: int = 0) -> np.ndarray:
    """Choose which alerts to embed in the dashboard.

    Taking simply the top N by confidence produces a queue that is 100% true
    positives on this corpus (precision@k stays at 1.000 well past k=4000), which
    would make the dashboard's false-positive view empty and quietly flatter the
    system. So the sample is stratified: the head of the queue, plus a slice of every
    severity band, plus a deliberate quota of known false positives. The result is a
    queue that shows the failure modes alongside the successes.

    ``truth`` may be omitted when no labels exist (a live capture), in which case only
    the confidence head and the severity strata are used.
    """
    rng = np.random.default_rng(seed)
    order = result.alert_rows
    chosen: list[int] = list(order[:top].tolist())

    for sev in ("critical", "high", "medium", "low"):
        idx = np.flatnonzero(result.is_alert & (result.severity == sev))
        if idx.size:
            take = idx if idx.size <= per_severity else rng.choice(
                idx, per_severity, replace=False)
            chosen.extend(int(i) for i in np.atleast_1d(take))

    if truth is not None and false_positives > 0:
        fp = np.flatnonzero(result.is_alert & ~truth)
        if fp.size:
            # Highest-confidence false positives first: those are the ones that would
            # actually cost an analyst time, so they are the ones worth showing.
            fp = fp[np.argsort(-result.fused_score[fp])][:false_positives]
            chosen.extend(int(i) for i in fp)

    seen, out = set(), []
    for i in chosen:
        if i not in seen:
            seen.add(i)
            out.append(i)
    arr = np.array(out, dtype=int)
    return arr[np.argsort(-result.fused_score[arr], kind="stable")]


def print_summary(report: dict) -> None:
    """Terminal rendering of the headline evaluation numbers."""
    d = report["dataset"]
    print(f"\nDataset : {d['name']}  ({d['source']})")
    print(f"          train {d['train_rows']:,} rows / test {d['test_rows']:,} rows")
    if d["is_simulated"]:
        print("          *** SIMULATED CORPUS -- see README, these are not "
              "NSL-KDD numbers ***")
    print(f"\nBinary detection on the test split (attack vs normal)")
    print(f"  {'configuration':<26} {'prec':>7} {'recall':>7} {'F1':>7} "
          f"{'FPR':>7} {'acc':>7}")
    for name in ("signature_only", "supervised_only", "anomaly_only",
                 "signature_or_supervised", "signature_or_anomaly", "hybrid"):
        m = report["ablation"][name]
        mark = " <" if name == "hybrid" else ""
        print(f"  {name:<26} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['false_positive_rate']:>7.4f} "
              f"{m['accuracy']:>7.4f}{mark}")
    mc = report["multiclass"]
    print(f"\nAttack-class assignment: accuracy {mc['accuracy']:.4f}, "
          f"macro F1 {mc['macro_f1']:.4f}")
    print(mc["confusion_text"])
    print(f"  {'class':<8} {'precision':>9} {'recall':>7} {'F1':>7} {'support':>8}")
    for r in mc["per_class"]:
        print(f"  {r['class']:<8} {r['precision']:>9.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['support']:>8}")
    print(f"\nAlert queue: {report['queue']['alerts']:,} alerts "
          f"({report['queue']['alert_rate']:.1%} of connections)")
    for s in report["severity"]:
        print(f"  {s['severity']:<9} {s['alerts']:>6,} alerts, "
              f"precision {s['precision']:.3f}")
    print(f"\nThroughput : {report['performance']['rows_per_second']:,.0f} "
          f"connections/second (detection only)")




