"""CICIDS2017 ingestion (secondary corpus).

NSL-KDD is the primary dataset because its 41 features are already
connection-level and its attack taxonomy maps cleanly onto the four traffic types
the brief asks for. CICIDS2017 is supported as a cross-check on modern,
packet-derived flow statistics: it is bidirectional-flow based, has ~78 numeric
features produced by CICFlowMeter, and contains its own scanning, DoS and
brute-force days.

Drop any of the ``MachineLearningCVE/*.csv`` files into ``data/raw/cicids/`` and
run ``python -m nids.cli evaluate --dataset cicids``.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from ..config import RAW_DIR

CICIDS_DIR = os.path.join(RAW_DIR, "cicids")

# CICIDS2017 label -> our coarse category.
LABEL_MAP = {
    "benign": "normal",
    "portscan": "probe",
    "dos hulk": "dos", "dos goldeneye": "dos", "dos slowloris": "dos",
    "dos slowhttptest": "dos", "ddos": "dos", "heartbleed": "dos",
    "ftp-patator": "r2l", "ssh-patator": "r2l",
    "web attack  brute force": "r2l", "web attack  xss": "r2l",
    "web attack  sql injection": "r2l", "infiltration": "u2r", "bot": "u2r",
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns]
    return df


def files_present() -> bool:
    return bool(glob.glob(os.path.join(CICIDS_DIR, "*.csv")))


def load_cicids(max_rows_per_file: int | None = 200_000, seed: int = 0):
    """Return ``(df, provenance)`` with ``category`` and ``is_attack`` columns.

    Raises FileNotFoundError if no CSVs are present -- unlike NSL-KDD there is no
    synthetic fallback for this corpus, because it is an optional cross-check.
    """
    paths = sorted(glob.glob(os.path.join(CICIDS_DIR, "*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No CICIDS2017 CSVs found in {CICIDS_DIR}. See README section "
            "'Optional: CICIDS2017'."
        )
    frames = []
    for p in paths:
        df = _clean_columns(pd.read_csv(p, low_memory=False))
        if max_rows_per_file and len(df) > max_rows_per_file:
            df = df.sample(max_rows_per_file, random_state=seed)
        df["__source_file"] = os.path.basename(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in CICIDS CSVs; got {list(df.columns)[:8]}")
    raw_label = df["label"].astype(str).str.strip().str.lower()
    df["label"] = raw_label
    df["category"] = raw_label.map(LABEL_MAP).fillna("unknown")
    df["is_attack"] = (df["category"] != "normal").astype(int)

    # CICFlowMeter emits inf for rate columns on zero-duration flows.
    num_cols = [c for c in df.columns
                if c not in ("label", "category", "is_attack", "__source_file")
                and pd.api.types.is_numeric_dtype(df[c])]
    df[num_cols] = (df[num_cols].replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0).astype(np.float64))

    prov = {
        "source": "real",
        "name": "CICIDS2017 (CICFlowMeter CSVs)",
        "train_path": CICIDS_DIR,
        "test_path": CICIDS_DIR,
        "note": f"Loaded {len(paths)} CSV file(s) from {CICIDS_DIR}.",
        "n_rows": int(len(df)),
        "numeric_features": num_cols,
        "files": [os.path.basename(p) for p in paths],
    }
    unknown = sorted(set(df.loc[df.category == "unknown", "label"]))
    if unknown:
        prov["unmapped_labels"] = unknown
    return df, prov


def stratified_train_test(df: pd.DataFrame, test_fraction: float = 0.3,
                          seed: int = 0):
    """Chronology-free stratified split; CICIDS ships as day-long captures."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _cat, grp in df.groupby("category"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1 - test_fraction)))
        train_idx.append(idx[:cut])
        test_idx.append(idx[cut:])
    tr = df.loc[np.concatenate(train_idx)].reset_index(drop=True)
    te = df.loc[np.concatenate(test_idx)].reset_index(drop=True)
    return tr, te

