"""Signature / rule layer.

This is the deterministic half of the hybrid detector. Every rule is a Boolean
expression over connection features with a fixed threshold, chosen from the
mechanics of the attack rather than fitted to the data -- which is exactly why it
is worth keeping alongside the models: it is auditable, it cannot drift, and its
alerts carry an explanation an analyst can verify by hand in one line.

The trade-off is equally explicit: a rule only fires on behaviour someone already
characterised, so the rule layer is expected to miss novel attacks. That is the
anomaly layer's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import RULE_THRESHOLDS as T

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Rule:
    rule_id: str
    name: str
    category: str                     # dos / probe / r2l / u2r
    severity: str
    rationale: str                    # why this behaviour indicates the attack
    condition: Callable[[pd.DataFrame], np.ndarray]
    evidence: list[str] = field(default_factory=list)
    targets: str = ""                 # attack families this is written for

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        mask = np.asarray(self.condition(df), dtype=bool)
        if mask.shape != (len(df),):
            raise ValueError(f"rule {self.rule_id} returned wrong shape")
        return mask


def _n(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)
    return np.zeros(len(df))


def _s(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        return df[col].astype(str).str.strip().str.lower().to_numpy(dtype=object)
    return np.full(len(df), "", dtype=object)


HALF_OPEN = ("s0", "s1", "s2", "s3")
REJECTED = ("rej", "rsto", "rstr")

RULES: list[Rule] = [
    # ------------------------------ DoS -------------------------------- #
    Rule(
        rule_id="R01", name="TCP SYN flood", category="dos", severity="critical",
        targets="neptune, processtable",
        rationale=(f"A backlog-exhaustion flood leaves connections half-open: the "
                   f"TCP state stays SYN-received, so serror_rate approaches 1 "
                   f"while the connection count to one host exceeds "
                   f"{T['syn_flood_min_count']}."),
        condition=lambda d: (
            np.isin(_s(d, "flag"), HALF_OPEN)
            & (np.maximum(_n(d, "serror_rate"), _n(d, "srv_serror_rate"))
               >= T["syn_flood_serror_rate"])
            & (_n(d, "count") >= T["syn_flood_min_count"])
        ),
        evidence=["flag", "serror_rate", "srv_serror_rate", "count",
                  "dst_host_serror_rate"],
    ),
    Rule(
        rule_id="R02", name="ICMP echo amplification flood", category="dos",
        severity="critical", targets="smurf, pod",
        rationale=(f"Smurf-style amplification sends oversized ICMP echo replies "
                   f"(>{T['icmp_flood_min_src_bytes']} bytes) at a saturating "
                   f"rate, with no TCP state to inspect."),
        condition=lambda d: (
            (_s(d, "protocol_type") == "icmp")
            & (_n(d, "src_bytes") >= T["icmp_flood_min_src_bytes"])
            & (_n(d, "count") >= 200)
        ),
        evidence=["protocol_type", "service", "src_bytes", "count", "srv_count"],
    ),
    Rule(
        rule_id="R03", name="Application-layer request flood", category="dos",
        severity="high", targets="back, apache2",
        rationale=("A well-formed HTTP flood completes its handshakes, so error "
                   "rates stay at zero. What gives it away is a high, perfectly "
                   "single-service connection rate carrying unusually large "
                   "requests."),
        condition=lambda d: (
            (_s(d, "service").astype(str) == "http")
            & (_n(d, "count") >= 60)
            & (_n(d, "same_srv_rate") >= 0.90)
            & (_n(d, "src_bytes") >= 1500)
        ),
        evidence=["service", "count", "same_srv_rate", "src_bytes", "duration"],
    ),
    Rule(
        rule_id="R04", name="Malformed IP fragmentation", category="dos",
        severity="high", targets="teardrop, pod",
        rationale=("Overlapping or truncated fragments are never produced by a "
                   "healthy stack; one bad fragment is enough to alert on."),
        condition=lambda d: _n(d, "wrong_fragment") >= T["malformed_min_wrong_fragment"],
        evidence=["protocol_type", "wrong_fragment", "src_bytes", "count"],
    ),
    Rule(
        rule_id="R05", name="Looped source/destination (LAND)", category="dos",
        severity="high", targets="land",
        rationale=("Source address equal to destination address is spoofing by "
                   "definition and crashes some stacks."),
        condition=lambda d: _n(d, "land") >= 1,
        evidence=["land", "protocol_type", "service", "flag"],
    ),
]

RULES += [
    # ----------------------- Probe / reconnaissance --------------------- #
    Rule(
        rule_id="R06", name="Vertical port scan", category="probe",
        severity="high", targets="portsweep, satan, mscan",
        rationale=(f"One host touched across many different services "
                   f"(diff_srv_rate >= {T['portscan_diff_srv_rate']}) with a long "
                   f"connection history but few repeats of any single service is "
                   f"port enumeration, not normal client behaviour."),
        condition=lambda d: (
            (_n(d, "diff_srv_rate") >= T["portscan_diff_srv_rate"])
            & (_n(d, "dst_host_count") >= T["portscan_min_dst_host_count"])
            & (_n(d, "dst_host_same_srv_rate") <= 0.35)
        ),
        evidence=["diff_srv_rate", "dst_host_count", "dst_host_diff_srv_rate",
                  "dst_host_same_srv_rate", "dst_host_srv_count"],
    ),
    Rule(
        rule_id="R07", name="Rejected-connection scan", category="probe",
        severity="medium", targets="satan, mscan, portsweep",
        rationale=(f"Closed ports answer with RST. A source whose connections are "
                   f"rejected at least {T['portscan_rerror_rate']:.0%} of the time "
                   f"is enumerating rather than using services."),
        condition=lambda d: (
            (np.maximum(_n(d, "rerror_rate"), _n(d, "srv_rerror_rate"))
             >= T["portscan_rerror_rate"])
            & (_n(d, "count") >= 3)
        ),
        evidence=["flag", "rerror_rate", "srv_rerror_rate", "count",
                  "dst_host_rerror_rate"],
    ),
    Rule(
        rule_id="R08", name="Horizontal host sweep", category="probe",
        severity="high", targets="ipsweep, nmap",
        rationale=(f"The same service probed across many different hosts "
                   f"(srv_diff_host_rate >= {T['hostsweep_srv_diff_host_rate']}) "
                   f"is address-space discovery."),
        condition=lambda d: (
            (_n(d, "srv_diff_host_rate") >= T["hostsweep_srv_diff_host_rate"])
            & (_n(d, "dst_host_srv_diff_host_rate") >= 0.40)
        ),
        evidence=["service", "srv_diff_host_rate", "dst_host_srv_diff_host_rate",
                  "srv_count", "dst_host_srv_count"],
    ),
    # ------------------- R2L: brute force / remote misuse ---------------- #
    Rule(
        rule_id="R09", name="Credential brute force", category="r2l",
        severity="high", targets="guess_passwd, ftp-patator, ssh-patator",
        rationale=(f"Repeated authentication failures within one connection "
                   f"(num_failed_logins >= {T['bruteforce_min_failed_logins']}) is "
                   f"password guessing. This is content-level evidence: the "
                   f"traffic volume is indistinguishable from a legitimate login."),
        condition=lambda d: _n(d, "num_failed_logins") >= T["bruteforce_min_failed_logins"],
        evidence=["service", "num_failed_logins", "logged_in", "is_guest_login",
                  "duration"],
    ),
    Rule(
        rule_id="R10", name="Guest session with file activity", category="r2l",
        severity="medium", targets="warezclient, warezmaster, ftp_write",
        rationale=("An anonymous/guest login that then creates files or triggers "
                   "hot indicators is using a public service as a drop point."),
        condition=lambda d: (
            (_n(d, "is_guest_login") >= 1)
            & ((_n(d, "hot") >= 2) | (_n(d, "num_file_creations") >= 1))
        ),
        evidence=["service", "is_guest_login", "hot", "num_file_creations",
                  "src_bytes"],
    ),
    # --------------------- U2R: privilege escalation --------------------- #
    Rule(
        rule_id="R11", name="Root shell obtained", category="u2r",
        severity="critical", targets="buffer_overflow, rootkit, xterm",
        rationale=("A root shell or an su attempt inside a monitored session is "
                   "the definition of privilege escalation. Very low volume, very "
                   "high impact -- exactly the case a rule should own rather than "
                   "a model trained on 50 examples."),
        condition=lambda d: (_n(d, "root_shell") >= 1) | (_n(d, "su_attempted") >= 1),
        evidence=["service", "root_shell", "su_attempted", "num_root", "hot",
                  "duration"],
    ),
    Rule(
        rule_id="R12", name="Root-level file manipulation", category="u2r",
        severity="high", targets="rootkit, loadmodule, ps",
        rationale=("Root-owned file operations combined with new file creation in "
                   "one session is post-exploitation persistence activity."),
        condition=lambda d: (_n(d, "num_root") >= 1) & (_n(d, "num_file_creations") >= 1),
        evidence=["num_root", "num_file_creations", "num_shells", "hot",
                  "num_access_files"],
    ),
]

RULES_BY_ID = {r.rule_id: r for r in RULES}


def _fmt(value) -> str:
    if isinstance(value, (int, float, np.floating, np.integer)):
        v = float(value)
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 and abs(v) < 1e15 else f"{v:.3f}"
    return str(value)


class SignatureEngine:
    """Evaluate all rules over a connection frame."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else RULES

    def run(self, df: pd.DataFrame) -> dict:
        """Return masks plus per-row rule hits.

        ``hits[i]`` is the list of rule_ids that fired for row i, ordered by
        descending severity so ``hits[i][0]`` is the headline rule.
        """
        masks = {r.rule_id: r.evaluate(df) for r in self.rules}
        n = len(df)
        hits: list[list[str]] = [[] for _ in range(n)]
        for rule in sorted(self.rules, key=lambda r: -SEVERITY_RANK[r.severity]):
            for i in np.flatnonzero(masks[rule.rule_id]):
                hits[int(i)].append(rule.rule_id)
        fired = np.array([bool(h) for h in hits])
        severity = np.array([
            RULES_BY_ID[h[0]].severity if h else "none" for h in hits], dtype=object)
        category = np.array([
            RULES_BY_ID[h[0]].category if h else "normal" for h in hits], dtype=object)
        return {"masks": masks, "hits": hits, "fired": fired,
                "severity": severity, "category": category}

    def explain(self, df: pd.DataFrame, row: int, rule_id: str) -> dict:
        """Human-readable justification for one (row, rule) pair."""
        rule = RULES_BY_ID[rule_id]
        observed = {}
        for col in rule.evidence:
            if col in df.columns:
                observed[col] = _fmt(df.iloc[row][col])
        return {
            "rule_id": rule.rule_id,
            "rule": rule.name,
            "category": rule.category,
            "severity": rule.severity,
            "why": rule.rationale,
            "targets": rule.targets,
            "observed": observed,
        }

    def coverage(self, df: pd.DataFrame, truth_column: str = "category") -> pd.DataFrame:
        """Per-rule fire count and precision against ground truth, for the report."""
        result = self.run(df)
        rows = []
        truth = df[truth_column].to_numpy(dtype=object) if truth_column in df else None
        for rule in self.rules:
            mask = result["masks"][rule.rule_id]
            fired = int(mask.sum())
            row = {"rule_id": rule.rule_id, "rule": rule.name,
                   "category": rule.category, "severity": rule.severity,
                   "fired": fired}
            if truth is not None and fired:
                sel = truth[mask]
                row["on_attack"] = int(np.sum(sel != "normal"))
                row["on_benign"] = int(np.sum(sel == "normal"))
                row["precision"] = row["on_attack"] / fired
                row["exact_category"] = int(np.sum(sel == rule.category)) / fired
            else:
                row.update({"on_attack": 0, "on_benign": 0, "precision": float("nan"),
                            "exact_category": float("nan")})
            rows.append(row)
        return pd.DataFrame(rows)



