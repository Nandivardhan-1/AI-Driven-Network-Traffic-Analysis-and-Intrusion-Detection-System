"""Hybrid fusion: three detectors, one prioritised alert queue.

Each layer answers a different question and fails in a different way:

===============  ==========================================  ==========================
layer            what it can find                            how it fails
===============  ==========================================  ==========================
signature        behaviour someone characterised, exactly     silent on anything novel
supervised       families present in the training labels      inherits label scarcity
anomaly          anything far from benign                     no idea *what* it found
===============  ==========================================  ==========================

Fusion is deliberately asymmetric. **Detection is a union**: any layer may raise an
alert, because suppressing a layer's finding to protect precision throws away the
only thing that layer was included for. **Priority is a product**: the confidence
score is a noisy-OR over the layers' weighted votes, so two layers agreeing scores
far above either alone, and a rule hit that both models confidently rate as normal
is still reported but drops to the bottom of the queue with its lack of
corroboration stated on the alert.

That distinction matters operationally. An analyst queue is finite; the cost of a
false positive is the time spent on it, not its existence. Ranking by corroboration
gives the analyst a queue where the top is dense with real incidents, without ever
silently discarding a detection.
"""
from __future__ import annotations

import pickle
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import FUSION, MODEL_DIR, RANDOM_SEED, SEVERITY_ORDER
from ..features import FeatureSpace
from .anomaly import AnomalyDetector
from .signatures import RULES_BY_ID, SEVERITY_RANK, SignatureEngine
from .supervised import SupervisedClassifier

# How much each layer's vote is trusted in the noisy-OR. The signature layer is
# weighted highest because its evidence is verifiable by hand; the anomaly layer
# lowest because it is the least precise of the three.
LAYER_WEIGHT = {"signature": 1.00, "supervised": 0.90, "anomaly": 0.60}

RULE_SEVERITY_SCORE = {"critical": 0.95, "high": 0.85, "medium": 0.65, "low": 0.50}


def _band(score: float) -> str:
    for cut, name in FUSION["severity_bands"]:
        if score >= cut:
            return name
    return "low"


@dataclass
class DetectionResult:
    """Everything the three layers produced, aligned row-for-row with the input."""
    n_rows: int
    rule_hits: list[list[str]]
    rule_score: np.ndarray
    anomaly_score: np.ndarray
    anomaly_novelty: np.ndarray
    anomaly_alert: np.ndarray
    supervised_proba: np.ndarray
    supervised_category: np.ndarray
    supervised_attack_prob: np.ndarray
    supervised_alert: np.ndarray
    fused_score: np.ndarray
    severity: np.ndarray
    category: np.ndarray
    is_alert: np.ndarray
    layers: list[list[str]]
    corroboration: np.ndarray
    classes: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)

    @property
    def alert_rows(self) -> np.ndarray:
        """Alerted row indices, highest fused score first."""
        idx = np.flatnonzero(self.is_alert)
        return idx[np.argsort(-self.fused_score[idx], kind="stable")]

    def summary(self) -> dict:
        sev, cnt = np.unique(self.severity[self.is_alert], return_counts=True)
        return {
            "rows": int(self.n_rows),
            "alerts": int(self.is_alert.sum()),
            "alert_rate": float(self.is_alert.mean()) if self.n_rows else 0.0,
            "by_severity": dict(zip(sev.tolist(), cnt.tolist())),
            "by_layer": {name: int(sum(name in ls for ls in self.layers))
                         for name in ("signature", "supervised", "anomaly")},
            "timing": dict(self.timing),
        }


class HybridDetector:
    """Signature + anomaly + supervised, fitted together and fused per connection."""

    def __init__(self, seed: int = RANDOM_SEED, n_trees: int = 120,
                 n_iforest: int = 150) -> None:
        self.seed = seed
        self.space = FeatureSpace()
        self.signatures = SignatureEngine()
        self.anomaly: AnomalyDetector | None = None
        self.supervised: SupervisedClassifier | None = None
        self._n_trees, self._n_iforest = n_trees, n_iforest
        self.fit_info: dict = {}

    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, verbose: bool = True) -> "HybridDetector":
        """Fit the feature space and both learned layers on training data only."""
        t0 = time.perf_counter()
        self.space.fit(train)
        t_space = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.supervised = SupervisedClassifier(
            self.space, n_estimators=self._n_trees, seed=self.seed).fit(train)
        t_sup = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.anomaly = AnomalyDetector(
            self.space, n_estimators=self._n_iforest, seed=self.seed).fit(train)
        t_ano = time.perf_counter() - t0

        self.fit_info = {
            "train_rows": int(len(train)),
            "n_features": self.space.n_features,
            "benign_rows_for_anomaly": self.anomaly.n_benign_,
            "anomaly_threshold": self.anomaly.threshold_,
            "anomaly_percentile": self.anomaly.percentile,
            "supervised_classes": list(self.supervised.classes_),
            "supervised_oob_accuracy": self.supervised.oob_score_,
            "seconds": {"features": t_space, "supervised": t_sup, "anomaly": t_ano},
        }
        if verbose:
            print(f"  feature space : {self.space.n_features} features "
                  f"({t_space:.1f}s)")
            print(f"  supervised    : {self._n_trees} trees, OOB accuracy "
                  f"{self.supervised.oob_score_:.4f} ({t_sup:.1f}s)")
            print(f"  anomaly       : {self._n_iforest} trees on "
                  f"{self.anomaly.n_benign_} benign rows, alert threshold "
                  f"{self.anomaly.threshold_:.4f} ({t_ano:.1f}s)")
        return self

    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> DetectionResult:
        if self.supervised is None or self.anomaly is None:
            raise RuntimeError("HybridDetector must be fitted first")
        n = len(df)

        t0 = time.perf_counter()
        sig = self.signatures.run(df)
        t_sig = time.perf_counter() - t0

        t0 = time.perf_counter()
        sup = self.supervised.predict(df)
        t_sup = time.perf_counter() - t0

        t0 = time.perf_counter()
        ano = self.anomaly.predict(df)
        t_ano = time.perf_counter() - t0

        rule_score = np.array([
            RULE_SEVERITY_SCORE[RULES_BY_ID[h[0]].severity] if h else 0.0
            for h in sig["hits"]])

        # Anomaly contribution is measured *above* the benign threshold, so a
        # connection that is merely typical contributes nothing at all.
        q = self.anomaly.percentile / 100.0
        ano_component = np.clip((ano["novelty"] - q) / max(1e-9, 1.0 - q), 0.0, 1.0)

        parts = np.column_stack([
            LAYER_WEIGHT["signature"] * rule_score,
            LAYER_WEIGHT["supervised"] * np.where(sup["alert"], sup["attack_prob"], 0.0),
            LAYER_WEIGHT["anomaly"] * np.where(ano["alert"], ano_component, 0.0),
        ])
        fused = 1.0 - np.prod(1.0 - np.clip(parts, 0.0, 1.0), axis=1)

        layers: list[list[str]] = []
        for i in range(n):
            ls = []
            if rule_score[i] > 0:
                ls.append("signature")
            if sup["alert"][i]:
                ls.append("supervised")
            if ano["alert"][i]:
                ls.append("anomaly")
            layers.append(ls)
        # dtype is pinned because an empty frame would otherwise yield a float64
        # array, and boolean indexing with it raises rather than returning nothing.
        is_alert = np.array([bool(ls) for ls in layers], dtype=bool)

        # ---- corroboration discount ------------------------------------ #
        # Detection is a union, but priority is not. A finding backed by one layer
        # alone is kept in the queue and scored down, by a factor reflecting how
        # much that layer can be trusted unaided: a rule is verifiable by hand, so
        # it keeps most of its score; an anomaly-only hit means "unlike normal, of
        # unknown nature", which is the weakest evidence in the system.
        discount = np.ones(n)
        for i in range(n):
            if len(layers[i]) >= 2:
                continue
            if layers[i] == ["signature"]:
                discount[i] = 0.62
            elif layers[i] == ["supervised"]:
                discount[i] = 0.55
            elif layers[i] == ["anomaly"]:
                discount[i] = 0.45
        fused = fused * discount

        severity = np.array([_band(s) if a else "info"
                             for s, a in zip(fused, is_alert)], dtype=object)
        corroboration = np.empty(n, dtype=object)
        for i in range(n):
            ls = layers[i]
            if not ls:
                corroboration[i] = "no layer flagged this connection"
            elif len(ls) >= 2:
                corroboration[i] = f"{len(ls)} of 3 layers agree ({', '.join(ls)})"
            elif ls == ["signature"]:
                corroboration[i] = (
                    "signature only -- the supervised model puts attack probability "
                    f"at {sup['attack_prob'][i]:.2f} and the connection sits at the "
                    f"{ano['novelty'][i]:.1%} percentile of benign traffic, below the "
                    f"{q:.0%} anomaly line, so the score is discounted for lack of "
                    "corroboration")
            elif ls == ["supervised"]:
                corroboration[i] = (
                    "supervised model only -- no rule matched and the connection is "
                    "inside the benign envelope, so this rests entirely on learned "
                    "class boundaries")
            else:
                corroboration[i] = (
                    "anomaly layer only -- unlike normal traffic but matching no "
                    "known attack pattern. This is the novel-behaviour path and is "
                    "the weakest single source of evidence, hence the low priority")

        # ---- category attribution -------------------------------------- #
        category = np.empty(n, dtype=object)
        for i in range(n):
            if sig["hits"][i]:
                category[i] = RULES_BY_ID[sig["hits"][i][0]].category
            elif sup["alert"][i]:
                category[i] = str(sup["category"][i])
            elif ano["alert"][i]:
                category[i] = "unknown"
            else:
                category[i] = "normal"

        return DetectionResult(
            n_rows=n, rule_hits=sig["hits"], rule_score=rule_score,
            anomaly_score=ano["score"], anomaly_novelty=ano["novelty"],
            anomaly_alert=ano["alert"], supervised_proba=sup["proba"],
            supervised_category=sup["category"],
            supervised_attack_prob=sup["attack_prob"],
            supervised_alert=sup["alert"], fused_score=fused, severity=severity,
            category=category, is_alert=is_alert, layers=layers,
            corroboration=corroboration, classes=list(sup["classes"]),
            timing={"signature_s": t_sig, "supervised_s": t_sup, "anomaly_s": t_ano,
                    "total_s": t_sig + t_sup + t_ano,
                    "rows_per_second": n / max(1e-9, t_sig + t_sup + t_ano)})

    # ------------------------------------------------------------------ #
    IDENTITY_FIELDS = ("src_ip", "dst_ip", "dst_port", "start_time",
                       "protocol_type", "service", "flag", "duration",
                       "src_bytes", "dst_bytes", "count", "srv_count")

    def explain_alerts(self, df: pd.DataFrame, result: DetectionResult,
                       rows: np.ndarray | None = None, limit: int | None = 50,
                       top_k: int = 5) -> list[dict]:
        """Build one fully-explained alert record per flagged connection.

        This is the implementation of requirement D: every record states what was
        flagged, which rule or which features caused it with the values observed,
        and a severity plus a confidence.
        """
        idx = result.alert_rows if rows is None else np.asarray(rows, dtype=int)
        if limit is not None:
            idx = idx[:limit]
        if idx.size == 0:
            return []

        sup_ex = self.supervised.explain(df, idx, top_k=top_k)
        ano_ex = self.anomaly.explain(df, idx, top_k=top_k)
        records = []
        for k, i in enumerate(idx.tolist()):
            row = df.iloc[i]
            identity = {f: (row[f].item() if hasattr(row[f], "item") else row[f])
                        for f in self.IDENTITY_FIELDS if f in df.columns}
            rules = [self.signatures.explain(df, i, rid)
                     for rid in result.rule_hits[i]]
            rec = {
                "row": int(i),
                "severity": str(result.severity[i]),
                "confidence": round(float(result.fused_score[i]), 4),
                "category": str(result.category[i]),
                "layers": list(result.layers[i]),
                "corroboration": str(result.corroboration[i]),
                "identity": identity,
                "rules": rules,
                "supervised": {
                    "predicted": sup_ex[k]["predicted"],
                    "probability": round(sup_ex[k]["probability"], 4),
                    "attack_probability": round(
                        float(result.supervised_attack_prob[i]), 4),
                    "class_probabilities": {
                        c: round(float(result.supervised_proba[i, j]), 4)
                        for j, c in enumerate(result.classes)},
                    "features": sup_ex[k]["features"],
                },
                "anomaly": {
                    "score": round(float(result.anomaly_score[i]), 4),
                    "threshold": round(float(self.anomaly.threshold_), 4),
                    "more_unusual_than": round(float(result.anomaly_novelty[i]), 4),
                    "alerted": bool(result.anomaly_alert[i]),
                    "features": ano_ex[k],
                },
            }
            for col in ("label", "category", "is_attack"):
                if col in df.columns:
                    rec.setdefault("ground_truth", {})[col] = str(row[col])
            rec["narrative"] = _narrative(rec)
            records.append(rec)
        return records

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict:
        """The rule catalogue is code, not fitted state.

        ``Rule.condition`` holds lambdas, which pickle cannot serialise, and
        serialising them would be wrong anyway: a saved model must pick up the
        current rule set when it is reloaded, not a frozen copy of an old one.
        """
        state = dict(self.__dict__)
        state["signatures"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.signatures = SignatureEngine()

    def save(self, path: str | None = None) -> str:
        """Persist the fitted detector. Pickle is used because the model classes are
        local to this repository -- there is no third-party serialiser involved."""
        path = path or os.path.join(MODEL_DIR, "hybrid.pkl")
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: str | None = None) -> "HybridDetector":
        path = path or os.path.join(MODEL_DIR, "hybrid.pkl")
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _narrative(rec: dict) -> str:
    """One plain-English paragraph explaining a single alert."""
    ident = rec["identity"]
    who = ""
    if "src_ip" in ident:
        who = f"{ident['src_ip']} -> {ident.get('dst_ip', '?')}"
        if "dst_port" in ident:
            who += f":{int(ident['dst_port'])}"
        who += " "
    svc = ident.get("service", "?")
    flag = ident.get("flag", "?")
    head = (f"{who}({svc}, state {flag}) flagged {rec['severity'].upper()} as "
            f"{rec['category']} with confidence {rec['confidence']:.2f}.")

    parts = []
    if rec["rules"]:
        r = rec["rules"][0]
        observed = ", ".join(f"{k}={v}" for k, v in r["observed"].items())
        extra = (f" A further {len(rec['rules']) - 1} rule(s) also matched."
                 if len(rec["rules"]) > 1 else "")
        parts.append(f"Rule {r['rule_id']} ({r['rule']}) matched: {r['why']} "
                     f"Observed: {observed}.{extra}")
    if "supervised" in rec["layers"]:
        s = rec["supervised"]
        top = "; ".join(f"{f['feature']}={f['observed']:.3g} "
                        f"({f['direction']} {s['predicted']} by "
                        f"{abs(f['contribution']):.3f})"
                        for f in s["features"][:3])
        parts.append(f"The supervised model assigns {s['probability']:.1%} to "
                     f"'{s['predicted']}'. Largest contributions: {top}.")
    if rec["anomaly"]["alerted"]:
        a = rec["anomaly"]
        top = "; ".join(f"{f['feature']}={f['observed']:.3g} "
                        f"({f['z']:+.1f} SD from the benign mean)"
                        for f in a["features"][:3])
        parts.append(f"The anomaly layer scores this more unusual than "
                     f"{a['more_unusual_than']:.1%} of benign training traffic. "
                     f"Most isolating features: {top}.")
    parts.append(f"Corroboration: {rec['corroboration']}.")
    return head + " " + " ".join(parts)






