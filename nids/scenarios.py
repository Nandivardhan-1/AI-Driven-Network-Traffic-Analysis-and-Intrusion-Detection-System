"""The five mandatory demonstration scenarios (brief section 6), as code.

Each scenario is a function taking the fitted detector and the test split and
returning a dict that both the CLI and the HTML dashboard render. They are written
to be *shown*, in order, during the video: each one prints the numbers, then one
fully-explained example alert, then the honest limitation it reveals.

Scenario 5 is deliberately not a success story. It is selected automatically as the
worst-performing attack family that was absent from the training labels, which is
the failure this architecture cannot design its way out of.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, SEVERITY_ORDER
from .detect.explain import format_alert, format_alert_line, alert_table_header
from .detect.hybrid import HybridDetector
from .ml.metrics import binary_scores


def _slice_scores(result, mask: np.ndarray, truth: np.ndarray) -> dict:
    """Detection statistics restricted to a subset of connections."""
    sub_truth, sub_pred = truth[mask], result.is_alert[mask]
    s = binary_scores(sub_truth.astype(int), sub_pred.astype(int))
    return {"connections": int(mask.sum()),
            "attacks": int(sub_truth.sum()),
            "alerts": int(sub_pred.sum()),
            "recall": round(float(s["recall"]), 4),
            "precision": round(float(s["precision"]), 4),
            "false_positive_rate": round(float(s["false_positive_rate"]), 4)}


def _rule_breakdown(result, mask: np.ndarray) -> dict:
    counts: dict[str, int] = {}
    for i in np.flatnonzero(mask):
        for rid in result.rule_hits[i]:
            counts[rid] = counts.get(rid, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _pick(result, mask: np.ndarray, n: int = 1) -> np.ndarray:
    """Highest-confidence alerted rows inside a mask."""
    idx = np.flatnonzero(mask & result.is_alert)
    if idx.size == 0:
        return idx
    return idx[np.argsort(-result.fused_score[idx])][:n]


def _rules_phrase(counts: dict, top: int = 4) -> str:
    """'R06 (vertical port scan, 113), R04 (...)' -- built from measured counts.

    Every narrative number in this module is derived here or in the scenario body
    rather than written into the prose. An earlier version hard-coded the figures
    from one corpus, and they silently became false the moment the detector was run
    on another one; a finding that cannot follow its own measurements is worse than
    no finding.
    """
    from .detect.signatures import RULES_BY_ID
    items = list(counts.items())[:top]
    if not items:
        return "no rule at all"
    return ", ".join(f"{rid} ({RULES_BY_ID[rid].name.lower()}, {n:,})"
                     for rid, n in items)


def scenario_1_baseline(det: HybridDetector, df: pd.DataFrame, result,
                        truth: np.ndarray) -> dict:
    """Normal traffic only: does the system stay quiet?"""
    benign = ~truth
    fp = benign & result.is_alert
    # Which benign sub-population is producing them, by the features that fired.
    layer_counts = {name: int(sum(name in result.layers[i]
                                  for i in np.flatnonzero(fp)))
                    for name in ("signature", "supervised", "anomaly")}
    worst = np.flatnonzero(fp)
    worst = worst[np.argsort(-result.fused_score[worst])][:3] if worst.size else worst
    rules = _rule_breakdown(result, fp)
    sev = {k: int(v) for k, v in
           zip(*np.unique(result.severity[fp], return_counts=True))}
    n_fp, n_benign = int(fp.sum()), int(benign.sum())
    rate = n_fp / max(1, n_benign)
    ranked_down = sev.get("medium", 0) + sev.get("low", 0)
    return {
        "id": 1,
        "title": "Normal traffic baseline",
        "question": "Run the detector on benign traffic only. How much noise does it make?",
        "stats": {
            "benign_connections": n_benign,
            "false_positives": n_fp,
            "false_positive_rate": round(float(rate), 4),
            "by_layer": layer_counts,
            "by_severity": sev,
        },
        "rules_involved": rules,
        "examples": det.explain_alerts(df, result, rows=worst, limit=None),
        "finding": (
            f"The system does not stay silent, and the brief's own wording anticipates "
            f"this: {n_fp:,} of {n_benign:,} benign connections raise something, a "
            f"false-positive rate of {rate:.2%} -- about 1 in "
            f"{round(1 / max(rate, 1e-9)):,}. The layers contribute "
            f"{layer_counts['signature']:,} rule hits, "
            f"{layer_counts['supervised']:,} supervised alerts and "
            f"{layer_counts['anomaly']:,} anomaly alerts (overlapping, so these sum to "
            f"more than the total). The rules doing it are {_rules_phrase(rules)}, and "
            f"the benign populations that trigger them are exactly the ones a real "
            f"network has: a monitoring system polling many hosts, a client retrying "
            f"closed ports, an anonymous FTP download, an administrator using su. No "
            f"threshold on these features separates them, because at connection level "
            f"they are not different -- the intent is, and intent is not in the record. "
            f"What the hybrid can do is rank: {ranked_down:,} of these land at medium or "
            f"low because no second layer corroborates them, so they sort below the "
            f"corroborated attacks rather than competing with them. "
            f"{sev.get('critical', 0):,} still reach critical, which is the residual "
            f"cost and is stated as such."),
    }


def scenario_2_recon(det: HybridDetector, df: pd.DataFrame, result,
                     truth: np.ndarray) -> dict:
    """Reconnaissance: port scans and host sweeps."""
    mask = (df["category"].astype(str) == "probe").to_numpy()
    fam = df["label"].astype(str).to_numpy(dtype=object)
    per_family = {f: round(float(result.is_alert[fam == f].mean()), 4)
                  for f in sorted(set(fam[mask]))}
    stats = _slice_scores(result, mask, truth)
    scan_rules = _rule_breakdown(result, mask)
    benign_rules = _rule_breakdown(result, ~truth)
    # How much of the benign rule noise comes from the very rules that carry this
    # scenario: the sensitivity and the false alarms are one mechanism, not two.
    shared = [r for r in list(scan_rules)[:3] if r in benign_rules]
    shared_hits = sum(benign_rules[r] for r in shared)
    fpr = float(result.is_alert[~truth].mean())
    perfect = [f for f, v in per_family.items() if v >= 0.999]
    return {
        "id": 2,
        "title": "Reconnaissance and port-scan detection",
        "question": "Can the system distinguish enumeration from ordinary client traffic?",
        "stats": stats | {"per_family": per_family},
        "rules_involved": scan_rules,
        "examples": det.explain_alerts(df, result, rows=_pick(result, mask, 2),
                                       limit=None),
        "finding": (
            f"This is the class the feature set is best suited to, and the result says "
            f"so: {stats['alerts']:,} of {stats['connections']:,} scan connections are "
            f"flagged ({stats['recall']:.1%} recall) across "
            f"{len(per_family)} families, {len(perfect)} of them at full recall "
            f"({', '.join(perfect) or 'none'}). Scanning is defined by the *spread* of "
            f"a source's connections rather than their content, and diff_srv_rate, "
            f"dst_host_diff_srv_rate and the rejected-connection rates measure that "
            f"spread directly, so a hand-written rule and a learned tree arrive at "
            f"nearly the same boundary. That is why recall generalises to a scanner "
            f"absent from training while it collapses for unseen r2l families in "
            f"scenario 5 -- the signal is geometric, not semantic. The cost of this "
            f"sensitivity is scenario 1: the same rules that carry this class "
            f"({', '.join(shared) or 'none'}) account for {shared_hits:,} rule hits on "
            f"benign traffic, because a monitoring sweep has the same geometry. High "
            f"recall on this class and a {fpr:.1%} benign alarm rate are the same fact "
            f"measured from two sides, not two independent results."),
    }


def scenario_3_dos(det: HybridDetector, df: pd.DataFrame, result,
                   truth: np.ndarray) -> dict:
    """Flooding, and what it does to the alert queue and to throughput."""
    mask = (df["category"].astype(str) == "dos").to_numpy()
    fam = df["label"].astype(str).to_numpy(dtype=object)
    per_family = {f: round(float(result.is_alert[fam == f].mean()), 4)
                  for f in sorted(set(fam[mask]))}
    alerts_from_dos = int((mask & result.is_alert).sum())
    total_alerts = int(result.is_alert.sum())
    # A flood produces one alert per connection. Grouping by (rule, service, flag)
    # is the cheapest correlation an operator would actually apply.
    key = list(zip(
        [",".join(result.rule_hits[i]) for i in np.flatnonzero(mask & result.is_alert)],
        df.loc[mask & result.is_alert, "service"].astype(str),
        df.loc[mask & result.is_alert, "flag"].astype(str)))
    distinct = len(set(key))
    stats = _slice_scores(result, mask, truth)
    # Where the recall that is missing actually goes, by family, measured rather
    # than asserted: the weakest three flooding families with real support.
    weak = sorted(((f, v) for f, v in per_family.items()
                   if int((fam == f).sum()) >= 10 and v < 0.9),
                  key=lambda kv: kv[1])[:3]
    weak_phrase = (", ".join(f"{f} {v:.1%} of {int((fam == f).sum()):,}"
                             for f, v in weak)
                   if weak else "no family with meaningful support")
    return {
        "id": 3,
        "title": "Denial-of-service detection and its cost",
        "question": ("Is the flood detected, and what does detecting it do to the "
                     "alert queue and to processing time?"),
        "stats": _slice_scores(result, mask, truth) | {
            "per_family": per_family,
            "share_of_all_alerts": round(alerts_from_dos / max(1, total_alerts), 4),
            "distinct_alert_signatures": distinct,
            "alerts_per_distinct_signature": round(alerts_from_dos / max(1, distinct), 1),
            "throughput_rows_per_second": round(
                result.timing["rows_per_second"], 1),
            "seconds_per_stage": {k: round(float(v), 4)
                                  for k, v in result.timing.items()
                                  if k.endswith("_s")},
        },
        "rules_involved": _rule_breakdown(result, mask),
        "examples": det.explain_alerts(df, result, rows=_pick(result, mask, 2),
                                       limit=None),
        "finding": (
            f"Flooding recall is {stats['recall']:.1%} over {stats['connections']:,} "
            f"connections, and where it succeeds the reason is mechanical rather than "
            f"statistical: a half-open flood pins serror_rate near 1 and drives count "
            f"into the hundreds, which no benign profile in the corpus reaches. The "
            f"cost is the queue. DoS traffic produces {alerts_from_dos:,} of the "
            f"{total_alerts:,} alerts in this run -- "
            f"{alerts_from_dos / max(1, total_alerts):.0%} of everything an analyst "
            f"would see -- across only {distinct} distinct alert signatures. Per-flow "
            f"alerting is the wrong granularity for a flood: correlation into one "
            f"incident per (rule, service, state) collapses it by a factor of "
            f"{alerts_from_dos / max(1, distinct):.0f}. The recall that is missing is "
            f"concentrated in specific families -- {weak_phrase} -- not spread evenly "
            f"across the class."),
    }


def scenario_4_credentials(det: HybridDetector, df: pd.DataFrame, result,
                           truth: np.ndarray) -> dict:
    """The additional attack type required by the brief: credential abuse.

    R2L (remote-to-local: password guessing, illicit remote access) and U2R
    (privilege escalation once inside) are treated together because they are two
    halves of one intrusion, and because they share the property that makes them
    hard here -- the evidence is in *content*, and a connection record only carries
    a handful of content counters.
    """
    cat = df["category"].astype(str).to_numpy(dtype=object)
    mask = (cat == "r2l") | (cat == "u2r")
    fam = df["label"].astype(str).to_numpy(dtype=object)
    rule_fired = result.rule_score > 0

    per_family = {}
    for f in sorted(set(fam[mask])):
        m = fam == f
        per_family[f] = {
            "support": int(m.sum()),
            "hybrid": round(float(result.is_alert[m].mean()), 4),
            "signature": round(float(rule_fired[m].mean()), 4),
            "supervised": round(float(result.supervised_alert[m].mean()), 4),
            "anomaly": round(float(result.anomaly_alert[m].mean()), 4),
        }
    # The specific claim this scenario makes: the rule layer is carrying these
    # classes, not the model. Quantify it instead of asserting it.
    gain = {}
    for f, v in per_family.items():
        gain[f] = round(v["hybrid"] - v["supervised"], 4)
    biggest = sorted(gain.items(), key=lambda kv: -kv[1])[:3]

    r2l_s = _slice_scores(result, cat == "r2l", truth)
    u2r_s = _slice_scores(result, cat == "u2r", truth)
    # The three families the model is worst on, among those with enough support to
    # mean anything, and what the anomaly layer manages across the whole slice.
    ranked = sorted((v for v in per_family.items() if v[1]["support"] >= 10),
                    key=lambda kv: kv[1]["supervised"])[:3]
    model_misses = ", ".join(f"{k} {v['supervised']:.1%}" for k, v in ranked)
    ano_best = max(per_family.items(), key=lambda kv: kv[1]["anomaly"], default=None)
    ano_note = (f"its best family here is {ano_best[0]} at "
                f"{ano_best[1]['anomaly']:.1%}" if ano_best else "it is silent")
    sig_only = sum(1 for v in per_family.values()
                   if v["signature"] > max(v["supervised"], v["anomaly"]))

    return {
        "id": 4,
        "title": "Credential brute force and privilege escalation (R2L / U2R)",
        "question": ("Beyond scanning and flooding, can the system catch an intruder "
                     "guessing a password and then escalating to root?"),
        "stats": _slice_scores(result, mask, truth) | {
            "r2l": r2l_s,
            "u2r": u2r_s,
            "per_family": per_family,
            "hybrid_gain_over_supervised": dict(biggest),
        },
        "rules_involved": _rule_breakdown(result, mask),
        "examples": det.explain_alerts(df, result, rows=_pick(result, mask, 3),
                                       limit=None),
        "finding": (
            f"This is where the hybrid architecture earns its cost, and also where the "
            f"system is weakest overall. R2L recall is {r2l_s['recall']:.1%} and U2R "
            f"{u2r_s['recall']:.1%} across {len(per_family)} families, and the "
            f"aggregate hides which layer is doing the work. These are the rarest "
            f"classes in the corpus, so the supervised layer has the least to learn "
            f"from: it recovers {model_misses}. The signature layer does not care "
            f"about class balance -- R09 counts failed logins, R10 flags guest sessions "
            f"that create files, and R11/R12 look for root_shell, su_attempted and "
            f"root-level file activity -- so on {sig_only} of these families it is the "
            f"strongest layer of the three. The largest hybrid-over-supervised gains "
            f"are {', '.join(f'{k} +{v:.0%}' for k, v in biggest)}. The anomaly layer "
            f"contributes little here ({ano_note}), because a brute-force login and a "
            f"root shell are statistically ordinary connections; where it is 0.000 that "
            f"is a null result and it is reported as one. Two honest caveats remain. "
            f"First, this is not detection of the attack but of its aftermath -- a "
            f"matched R11 means a root shell was already obtained. Second, the same "
            f"counters fire on an administrator legitimately using su, which is part of "
            f"the scenario-1 false-positive rate: the signature layer trades false "
            f"alarms on benign admin activity for recall on privilege escalation. On "
            f"this corpus that trade is worth making, but it is a trade, not a free "
            f"improvement."),
    }


def scenario_5_ambiguous(det: HybridDetector, df: pd.DataFrame, result,
                         truth: np.ndarray, novel: set[str] | None = None) -> dict:
    """The failure case. Selected automatically, not hand-picked.

    The brief asks for an ambiguous, borderline or missed case and the limitation it
    reveals. Rather than choose a flattering example, this picks the attack family
    with the *worst* hybrid recall among those absent from the training labels --
    the zero-day proxy. Nothing about the architecture can fix it, which is the point.
    """
    from .config import NOVEL_TEST_ATTACKS
    novel = novel if novel is not None else set(NOVEL_TEST_ATTACKS)
    fam = df["label"].astype(str).to_numpy(dtype=object)
    cats = df["category"].astype(str).to_numpy(dtype=object)
    rule_fired = result.rule_score > 0

    present = sorted({f for f in set(fam[truth]) if f in novel})
    table = []
    for f in present:
        m = fam == f
        table.append({"family": f, "category": str(cats[m][0]),
                      "support": int(m.sum()),
                      "hybrid_recall": round(float(result.is_alert[m].mean()), 4),
                      "signature_recall": round(float(rule_fired[m].mean()), 4),
                      "supervised_recall": round(float(result.supervised_alert[m].mean()), 4),
                      "anomaly_recall": round(float(result.anomaly_alert[m].mean()), 4)})
    table.sort(key=lambda r: (r["hybrid_recall"], -r["support"]))
    worst = table[0] if table else None

    known = np.array([f not in novel for f in fam])
    seen_recall = float(result.is_alert[truth & known].mean()) if (truth & known).any() else 0.0
    unseen_recall = float(result.is_alert[truth & ~known].mean()) if (truth & ~known).any() else 0.0

    # Show the misses themselves: the near-misses of the worst family, so the
    # explanation shows what the layers *did* see and why it was not enough.
    examples: list[dict] = []
    if worst is not None:
        m = (fam == worst["family"]) & ~result.is_alert
        idx = np.flatnonzero(m)
        if idx.size:
            idx = idx[np.argsort(-result.anomaly_novelty[idx])][:2]
            examples = det.explain_alerts(df, result, rows=idx, limit=None)

    return {
        "id": 5,
        "title": "Missed and ambiguous cases: attacks absent from training",
        "question": ("What does the system fail on, and is that failure fixable "
                     "within this design?"),
        "stats": {
            "recall_on_families_seen_in_training": round(seen_recall, 4),
            "recall_on_families_absent_from_training": round(unseen_recall, 4),
            "recall_gap": round(seen_recall - unseen_recall, 4),
            "worst_family": worst,
            "novel_families": table,
        },
        "rules_involved": {},
        "examples": examples,
        "finding": (
            f"Recall on attack families present in the training labels is "
            f"{seen_recall:.1%}; on families absent from them it is "
            f"{unseen_recall:.1%}. That gap of {seen_recall - unseen_recall:.0%} is the "
            f"single most important number in this evaluation, and it is the one a "
            f"headline accuracy figure hides."
            + (f" The worst case is '{worst['family']}' ({worst['category']}, "
               f"{worst['support']} connections): hybrid recall "
               f"{worst['hybrid_recall']:.1%}, with the signature layer at "
               f"{worst['signature_recall']:.1%}, the supervised layer at "
               f"{worst['supervised_recall']:.1%} and the anomaly layer at "
               f"{worst['anomaly_recall']:.1%}." if worst else "")
            + " Every layer fails for its own reason and the reasons compound rather "
              "than cancel: no rule was written for behaviour nobody had "
              "characterised, the classifier has no decision region for a class it "
              "never saw, and the anomaly layer -- the one layer that is supposed to "
              "cover exactly this -- is silent because these connections are not "
              "statistically unusual. They are ordinary-looking sessions on ordinary "
              "services carrying a hostile payload. At connection-record granularity "
              "they are genuinely indistinguishable from benign traffic, so this is "
              "not a tuning problem: no threshold on these features separates them. "
              "Closing it needs different evidence -- payload inspection, or "
              "host-level telemetry -- not a better model over the same features."),
    }


def scenario_6_pcap(det: HybridDetector, pcap_path: str,
                    max_packets: int | None = None) -> dict:
    """End-to-end on a real packet capture, not a pre-computed feature table.

    Not one of the five mandatory scenarios; included because the brief asks whether
    capture is reliable and complete (section 5), and the only way to answer that is
    to run the detector on flows this code derived from raw packets itself.
    """
    from .data.pcap import load_pcap_as_frame

    t0 = __import__("time").perf_counter()
    frame = load_pcap_as_frame(pcap_path, max_packets=max_packets)
    parse_s = __import__("time").perf_counter() - t0
    result = det.predict(frame)

    src = frame["src_ip"].astype(str).to_numpy(dtype=object) \
        if "src_ip" in frame.columns else np.array(["?"] * len(frame), dtype=object)
    by_source = {}
    for ip in sorted(set(src)):
        m = src == ip
        hits: dict[str, int] = {}
        for i in np.flatnonzero(m & result.is_alert):
            for rid in result.rule_hits[i]:
                hits[rid] = hits.get(rid, 0) + 1
        by_source[ip] = {
            "flows": int(m.sum()),
            "alerts": int((m & result.is_alert).sum()),
            "alert_rate": round(float(result.is_alert[m].mean()), 4),
            "top_severity": (max((result.severity[i] for i in
                                  np.flatnonzero(m & result.is_alert)),
                                 key=lambda s: SEVERITY_ORDER.index(s))
                             if (m & result.is_alert).any() else "none"),
            "rules": dict(sorted(hits.items(), key=lambda kv: -kv[1])),
        }

    # Narrative below is assembled from by_source rather than written by hand: the
    # demo capture's four hosts can land differently on a differently fitted
    # detector, and a finding that contradicts the table printed above it is worse
    # than no finding.
    def _who(ip: str) -> str:
        v = by_source[ip]
        return (f"{ip} ({v['alerts']}/{v['flows']} flows, top severity "
                f"{v['top_severity']})")

    quiet = [ip for ip, v in by_source.items() if v["alerts"] == 0]
    rule_backed = [ip for ip, v in by_source.items() if v["rules"]]
    model_only = [ip for ip, v in by_source.items()
                  if v["alerts"] and not v["rules"]]
    parts = []
    if quiet:
        parts.append("No alert at all is raised for "
                     + ", ".join(_who(ip) for ip in quiet) + ".")
    if rule_backed:
        parts.append("Rule-backed detections: " + "; ".join(
            f"{_who(ip)} via "
            + ", ".join(f"{r} x{n}" for r, n in by_source[ip]["rules"].items())
            for ip in rule_backed) + ".")
    if model_only:
        parts.append(
            "Flagged by the learned layers with no rule behind them: "
            + ", ".join(_who(ip) for ip in model_only)
            + ". Those are the interesting ones. They are detected, but they are "
            "detected weakly -- no signature matched, so nothing corroborates the "
            "models and the corroboration discount pushes them to the bottom of "
            "the queue. The structural reason is the record format: the "
            "dst_host_* features summarise behaviour per destination, and a sweep "
            "touches each destination once, so that history is empty at the moment "
            "each flow is scored. A scan that spreads across hosts defeats "
            "features that summarise behaviour per host, and the flow-level rules "
            "that would catch it have nothing to count.")

    return {
        "id": 6,
        "title": "Raw pcap: capture, flow assembly and detection end to end",
        "question": ("Given packets rather than a feature table, does the pipeline "
                     "still work -- and what does flow assembly lose?"),
        "stats": {
            "pcap": pcap_path,
            "flows": int(len(frame)),
            "parse_seconds": round(parse_s, 3),
            "alerts": int(result.is_alert.sum()),
            "by_source": by_source,
            "queue": result.summary(),
        },
        "rules_involved": _rule_breakdown(result, np.ones(len(frame), dtype=bool)),
        "examples": det.explain_alerts(df=frame, result=result,
                                       rows=result.alert_rows[:2], limit=None),
        "finding": (
            "The pipeline runs unchanged on flows assembled from raw packets, which "
            "is the useful part: the same rules and the same fitted models apply to "
            f"{len(frame)} flows reconstructed from the capture in {parse_s:.2f}s. "
            "What the capture shows is the limit of the record format rather than of "
            "the detector. " + " ".join(parts)
            + " That is an argument for correlating across destinations at a level "
              "above the flow record, which this design does not do."),
    }


# --------------------------------------------------------------------------- #
# Driver and terminal rendering
# --------------------------------------------------------------------------- #
def run_scenarios(det: HybridDetector, test: pd.DataFrame, result=None,
                  truth: np.ndarray | None = None,
                  novel: set[str] | None = None,
                  pcap_path: str | None = None) -> list[dict]:
    """Run every scenario against one already-computed detection pass."""
    if result is None:
        result = det.predict(test)
    if truth is None:
        truth = (test["category"].astype(str) != "normal").to_numpy()

    out = [
        scenario_1_baseline(det, test, result, truth),
        scenario_2_recon(det, test, result, truth),
        scenario_3_dos(det, test, result, truth),
        scenario_4_credentials(det, test, result, truth),
        scenario_5_ambiguous(det, test, result, truth, novel=novel),
    ]
    if pcap_path and os.path.exists(pcap_path):
        try:
            out.append(scenario_6_pcap(det, pcap_path))
        except Exception as exc:                                # pragma: no cover
            out.append({"id": 6, "title": "Raw pcap end to end",
                        "question": "", "stats": {"error": str(exc)},
                        "rules_involved": {}, "examples": [],
                        "finding": f"pcap scenario failed: {exc}"})
    return out


def _fmt_value(v, indent: int = 0) -> list[str]:
    pad = "    " + " " * indent
    lines = []
    if isinstance(v, dict):
        for k, sub in v.items():
            if isinstance(sub, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(_fmt_value(sub, indent + 2))
            else:
                lines.append(f"{pad}{k:<38} {sub}")
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                lines.append(f"{pad}- " + "  ".join(
                    f"{k}={item[k]}" for k in list(item)[:6]))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{v}")
    return lines


def print_scenario(sc: dict, colour: bool = True, examples: int = 1,
                   verbose: bool = True) -> None:
    """Render one scenario the way it should be walked through on camera."""
    bar = "=" * 78
    head = f"SCENARIO {sc['id']} -- {sc['title']}"
    print(f"\n{bar}\n{head}\n{bar}")
    if sc.get("question"):
        import textwrap
        print(textwrap.fill(sc["question"], 78, initial_indent="  ",
                            subsequent_indent="  "))
    print("\n  measurements")
    for line in _fmt_value(sc.get("stats", {})):
        print(line)
    if sc.get("rules_involved"):
        print("\n  rules that fired on this slice")
        for rid, n in list(sc["rules_involved"].items())[:12]:
            print(f"    {rid:<6} {n:>8,}")
    exs = sc.get("examples") or []
    if exs:
        print(f"\n  explained alert{'s' if min(examples, len(exs)) > 1 else ''} "
              f"({min(examples, len(exs))} of {len(exs)})")
        print("  " + alert_table_header())
        for rec in exs[:examples]:
            print("  " + format_alert_line(rec, colour))
        print()
        for rec in exs[:examples]:
            print(format_alert(rec, colour=colour, verbose=verbose))
            print()
    print("  finding")
    import textwrap
    print(textwrap.fill(sc["finding"], 76, initial_indent="    ",
                        subsequent_indent="    "))


def print_scenarios(scs: list[dict], colour: bool = True, examples: int = 1,
                    verbose: bool = True) -> None:
    for sc in scs:
        print_scenario(sc, colour=colour, examples=examples, verbose=verbose)


def save_scenarios(scs: list[dict], path: str | None = None) -> str:
    path = path or os.path.join(ARTIFACT_DIR, "scenarios.json")

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scs, fh, indent=2, default=_default)
    return path


