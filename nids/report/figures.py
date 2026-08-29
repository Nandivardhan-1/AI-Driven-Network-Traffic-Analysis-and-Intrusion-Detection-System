"""Figures for the report and the dashboard, drawn from the evaluation report only.

Every figure here is a function of ``evaluation.json``. Nothing is recomputed, so a
figure can never disagree with the number in the table beside it. Matplotlib is used
with the Agg backend because there is no display in the target environment.

Chart choices are deliberate rather than decorative:

* the ablation chart plots precision *and* recall as paired bars, because the whole
  argument for the hybrid is that it buys recall at a known precision cost, and a
  single-metric bar chart would hide the trade;
* the per-family chart is sorted by recall and colours the families absent from
  training separately, so the zero-day collapse is visible without reading labels;
* the ROC and PR curves are drawn from the stored curve arrays, with the operating
  point of the deployed thresholds marked -- a curve without the operating point
  invites the reader to imagine a threshold that was never used.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import FIGURE_DIR

# A restrained palette: one accent for the hybrid, grey for components, red for
# anything that represents a failure or a cost.
C_HYBRID = "#1f4e79"
C_COMPONENT = "#9bb7d4"
C_BAD = "#b3202c"
C_GOOD = "#2e7d5b"
C_GREY = "#8a8f98"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "figure.autolayout": False,
})


def _save(fig, name: str, outdir: str) -> str:
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_ablation(report: dict, outdir: str) -> str:
    """Precision and recall for each layer alone, the unions, and the hybrid."""
    order = ["signature_only", "supervised_only", "anomaly_only",
             "signature_or_anomaly", "signature_or_supervised", "hybrid"]
    labels = ["signature\nonly", "supervised\nonly", "anomaly\nonly",
              "signature\nOR anomaly", "signature\nOR supervised", "hybrid\n(all three)"]
    prec = [report["ablation"][k]["precision"] for k in order]
    rec = [report["ablation"][k]["recall"] for k in order]
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    b1 = ax.bar(x - w / 2, prec, w, label="precision", color=C_COMPONENT,
                edgecolor="white")
    b2 = ax.bar(x + w / 2, rec, w, label="recall", color=C_HYBRID,
                edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("score on the test split")
    ax.set_title("Layer ablation: what each layer contributes,\nmeasured on the "
                 "identical test split")
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), fontsize=8)
    return _save(fig, "fig1_ablation.png", outdir)


def fig_family_recall(report: dict, outdir: str) -> str:
    """Per-family recall, with families absent from training marked."""
    fams = sorted(report["families"], key=lambda r: r["hybrid_recall"])
    names = [f"{r['family']} ({r['category']}, n={r['support']})" for r in fams]
    vals = [r["hybrid_recall"] for r in fams]
    seen = [r["in_training"] for r in fams]
    colours = [C_GOOD if s else C_BAD for s in seen]

    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.24 * len(fams))))
    ax.barh(np.arange(len(fams)), vals, color=colours, edgecolor="white", height=0.72)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.015, 1.0), i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(np.arange(len(fams)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("hybrid recall")
    ax.set_title("Recall by attack family. Red = family absent from the training "
                 "labels")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_GOOD),
               plt.Rectangle((0, 0), 1, 1, color=C_BAD)]
    ax.legend(handles, ["seen in training", "absent from training"],
              frameon=False, loc="lower right", fontsize=8)
    return _save(fig, "fig2_family_recall.png", outdir)


def fig_layer_by_family(report: dict, outdir: str) -> str:
    """Which layer catches which family: the case for the hybrid, per class."""
    fams = sorted(report["families"],
                  key=lambda r: (r["category"], -r["support"]))
    keys = ["signature_recall", "supervised_recall", "anomaly_recall"]
    labels = ["signature", "supervised", "anomaly"]
    m = np.array([[f[k] for f in fams] for k in keys])

    fig, ax = plt.subplots(figsize=(max(7.6, 0.32 * len(fams)), 2.9))
    im = ax.imshow(m, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f["family"] for f in fams], rotation=90, fontsize=6.5)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if m[i, j] > 0.55 else "#333333")
    for j, f in enumerate(fams):
        if not f["in_training"]:
            ax.get_xticklabels()[j].set_color(C_BAD)
            ax.get_xticklabels()[j].set_fontweight("bold")
    ax.grid(False)
    ax.set_title("Recall per layer per family (red family names were absent from "
                 "training)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("recall", fontsize=8)
    return _save(fig, "fig3_layer_by_family.png", outdir)


def fig_curves(report: dict, outdir: str) -> str:
    """ROC for the two scored layers, and the hybrid precision-recall curve."""
    cur = report["curves"]
    auc = report["score_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))

    ax = axes[0]
    ax.plot(cur["supervised_roc"]["fpr"], cur["supervised_roc"]["tpr"],
            color=C_HYBRID, lw=1.6,
            label=f"supervised  AUC {auc['supervised_roc_auc']:.3f}")
    ax.plot(cur["anomaly_roc"]["fpr"], cur["anomaly_roc"]["tpr"],
            color=C_GREY, lw=1.6,
            label=f"anomaly  AUC {auc['anomaly_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.9, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC: the two scored layers")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.plot(cur["hybrid_pr"]["recall"], cur["hybrid_pr"]["precision"],
            color=C_HYBRID, lw=1.6, label="hybrid fused score")
    h = report["ablation"]["hybrid"]
    ax.scatter([h["recall"]], [h["precision"]], color=C_BAD, zorder=5, s=28,
               label=f"deployed thresholds\n({h['recall']:.3f}, {h['precision']:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hybrid precision-recall  (AUC {auc['hybrid_roc_auc']:.3f} ROC)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    return _save(fig, "fig4_curves.png", outdir)


def fig_confusion(report: dict, outdir: str) -> str:
    """Row-normalised multiclass confusion for the supervised layer."""
    cm = np.array(report["multiclass"]["confusion_matrix"], dtype=float)
    classes = report["multiclass"]["classes"]
    norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n{int(cm[i, j])}", ha="center",
                    va="center", fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else "#333333")
    ax.grid(False)
    ax.set_title("Supervised layer: class confusion\n(row-normalised, count below)")
    return _save(fig, "fig5_confusion.png", outdir)


def fig_severity_precision(report: dict, outdir: str) -> str:
    """Does the priority ranking work? Precision by severity band and precision@k."""
    sev = report["severity"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))

    ax = axes[0]
    names = [s["severity"] for s in sev]
    prec = [s["precision"] for s in sev]
    size = [s["alerts"] for s in sev]
    bars = ax.bar(names, prec, color=[C_HYBRID if p >= 0.9 else C_COMPONENT
                                      for p in prec], edgecolor="white")
    for b, n in zip(bars, size):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.3f}\nn={n:,}", ha="center", va="bottom",
                fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("precision of alerts in band")
    ax.set_title("Precision by severity band")

    ax = axes[1]
    pk = report["precision_at_k"]
    ax.plot([p["k"] for p in pk], [p["precision"] for p in pk], marker="o",
            ms=3.5, color=C_HYBRID, lw=1.5)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k (alerts reviewed, highest confidence first)")
    ax.set_ylabel("precision@k")
    ax.set_title("Queue quality: precision of the top k alerts")
    return _save(fig, "fig6_severity_precision.png", outdir)


def fig_cost(report: dict, outdir: str) -> str:
    """Where the time goes, and what the queue looks like: the section 5 evidence."""
    per = report["performance"]["per_stage_seconds"]
    stages = [(k[:-2], v) for k, v in per.items()
              if k.endswith("_s") and k != "total_s"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))

    ax = axes[0]
    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    total = sum(vals) or 1.0
    bars = ax.barh(names, vals, color=[C_BAD if v / total > 0.5 else C_COMPONENT
                                       for v in vals], edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2f}s ({v / total:.0%})",
                va="center", fontsize=7.5)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_xlabel("seconds for the whole test split")
    ax.set_title(f"Cost per stage  ({report['performance']['rows_per_second']:,.0f} "
                 f"conn/s overall)")

    ax = axes[1]
    q = report["queue"]["by_severity"]
    order = [s for s in ("critical", "high", "medium", "low") if s in q]
    vals = [q[s] for s in order]
    ax.bar(order, vals, color=[C_BAD, "#d97b45", "#e0c341", C_GREY][:len(order)],
           edgecolor="white")
    for x, v in zip(order, vals):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("alerts in the queue")
    ax.set_title(f"Alert queue composition ({report['queue']['alerts']:,} alerts "
                 f"from {report['queue']['rows']:,} connections)")
    return _save(fig, "fig7_cost_and_queue.png", outdir)


def fig_importances(report: dict, outdir: str) -> str:
    """Top features by impurity reduction, coloured by feature group."""
    imps = report["importances"][:16][::-1]
    names = [r["feature"] for r in imps]
    vals = [r["importance"] for r in imps]
    groups = [r.get("group", "?") for r in imps]
    uniq = sorted(set(groups))
    cmap = plt.get_cmap("tab10")
    cols = {g: cmap(i % 10) for i, g in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.26 * len(imps))))
    ax.barh(np.arange(len(imps)), vals, color=[cols[g] for g in groups],
            edgecolor="white", height=0.74)
    ax.set_yticks(np.arange(len(imps)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("mean impurity reduction (normalised)")
    ax.set_title("Feature importance in the supervised layer, by feature group")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[g]) for g in uniq]
    ax.legend(handles, uniq, frameon=False, fontsize=7, loc="lower right")
    return _save(fig, "fig8_importances.png", outdir)


def fig_scenario_summary(scenarios: list[dict], outdir: str) -> str:
    """One chart summarising the five mandatory scenarios side by side."""
    rows = []
    names = {1: "1 benign\nbaseline", 2: "2 recon /\nport scan",
             3: "3 denial\nof service", 4: "4 credential\nabuse",
             5: "5 unseen\nfamilies"}
    for sc in scenarios:
        st = sc.get("stats", {})
        if sc["id"] == 1:
            rows.append((names[1], st.get("false_positive_rate", 0.0),
                         "false-positive rate", C_BAD))
        elif sc["id"] in (2, 3, 4):
            rows.append((names[sc["id"]], st.get("recall", 0.0), "recall", C_GOOD))
        elif sc["id"] == 5:
            rows.append((names[5],
                         st.get("recall_on_families_absent_from_training", 0.0),
                         "recall", C_BAD))
    if not rows:
        rows = [("no data", 0.0, "", C_GREY)]

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows],
                  color=[r[3] for r in rows], edgecolor="white", width=0.62)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{r[1]:.3f}\n{r[2]}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("measured value")
    ax.set_title("The five mandatory scenarios, one number each "
                 "(red = a cost or a failure)")
    return _save(fig, "fig9_scenarios.png", outdir)


def build_all(report: dict, scenarios: list[dict] | None = None,
              outdir: str | None = None, verbose: bool = True) -> list[str]:
    """Draw every figure. Returns the list of paths written."""
    outdir = outdir or FIGURE_DIR
    os.makedirs(outdir, exist_ok=True)
    paths = []
    jobs = [("ablation", fig_ablation), ("family recall", fig_family_recall),
            ("layer x family", fig_layer_by_family), ("curves", fig_curves),
            ("confusion", fig_confusion),
            ("severity / precision@k", fig_severity_precision),
            ("cost and queue", fig_cost), ("importances", fig_importances)]
    for name, fn in jobs:
        try:
            p = fn(report, outdir)
            paths.append(p)
            if verbose:
                print(f"  {name:<24} -> {os.path.basename(p)}")
        except Exception as exc:                                # pragma: no cover
            print(f"  {name:<24} FAILED: {type(exc).__name__}: {exc}")
    if scenarios:
        try:
            p = fig_scenario_summary(scenarios, outdir)
            paths.append(p)
            if verbose:
                print(f"  {'scenario summary':<24} -> {os.path.basename(p)}")
        except Exception as exc:                                # pragma: no cover
            print(f"  scenario summary FAILED: {type(exc).__name__}: {exc}")
    return paths
