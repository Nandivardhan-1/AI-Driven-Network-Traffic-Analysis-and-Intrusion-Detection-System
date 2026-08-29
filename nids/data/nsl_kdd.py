"""NSL-KDD ingestion.

Loading order is deliberate and is reported to the user on every run:

1. ``data/raw/KDDTrain+.txt`` and ``data/raw/KDDTest+.txt`` if both exist -- the
   real benchmark, and what all headline numbers should be quoted from.
2. Otherwise the seeded synthetic corpus from ``nids.data.simulate``, so the repo
   is runnable with no downloads.

The provenance dict returned alongside the frames is threaded through to the CLI
banner, the HTML dashboard and the report so a reader can never mistake one source
for the other.
"""
from __future__ import annotations

import os

import pandas as pd

from ..config import (NSL_COLUMNS, NSL_TEST_FILE, NSL_TRAIN_FILE, RAW_DIR,
                      category_of)
from . import simulate


def _read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=NSL_COLUMNS, low_memory=False)
    # KDD'99 exports terminate labels with a period; NSL-KDD does not. Accept both.
    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".").str.lower()
    for col in NSL_COLUMNS:
        if col not in ("protocol_type", "service", "flag", "label"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("protocol_type", "service", "flag"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def real_files_present() -> bool:
    return all(os.path.exists(os.path.join(RAW_DIR, f))
               for f in (NSL_TRAIN_FILE, NSL_TEST_FILE))


def load_nsl_kdd(force_simulated: bool = False, n_train: int | None = None,
                 n_test: int | None = None):
    """Return ``(train_df, test_df, provenance)``.

    Both frames carry two derived label columns used everywhere downstream:
    ``category`` (normal/dos/probe/r2l/u2r) and ``is_attack`` (0/1).
    """
    if real_files_present() and not force_simulated:
        train = _read_raw(os.path.join(RAW_DIR, NSL_TRAIN_FILE))
        test = _read_raw(os.path.join(RAW_DIR, NSL_TEST_FILE))
        prov = {
            "source": "real",
            "name": "NSL-KDD (KDDTrain+ / KDDTest+)",
            "train_path": os.path.join(RAW_DIR, NSL_TRAIN_FILE),
            "test_path": os.path.join(RAW_DIR, NSL_TEST_FILE),
            "note": "Real benchmark files found in data/raw and used as-is.",
        }
    else:
        train, test = simulate.simulate_nsl_kdd(
            n_train=n_train or simulate.DEFAULT_TRAIN_ROWS,
            n_test=n_test or simulate.DEFAULT_TEST_ROWS,
        )
        prov = {
            "source": "simulated",
            "name": "SIMULATED NSL-KDD-schema corpus",
            "train_path": "<generated: nids.data.simulate>",
            "test_path": "<generated: nids.data.simulate>",
            "note": ("Real KDDTrain+.txt / KDDTest+.txt were not found in "
                     "data/raw. Figures are from the seeded synthetic corpus; "
                     "run scripts/fetch_nsl_kdd.py then re-run to use the real "
                     "benchmark."),
        }

    for df in (train, test):
        df["category"] = df["label"].map(category_of)
        df["is_attack"] = (df["category"] != "normal").astype(int)

    unknown = sorted(set(train.loc[train.category == "unknown", "label"]) |
                     set(test.loc[test.category == "unknown", "label"]))
    if unknown:
        prov["unmapped_labels"] = unknown

    prov["n_train"] = int(len(train))
    prov["n_test"] = int(len(test))
    prov["train_families"] = sorted(set(train["label"]))
    prov["test_families"] = sorted(set(test["label"]))
    prov["novel_in_test"] = sorted(set(test["label"]) - set(train["label"]))
    return train, test, prov


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Category x count table used in the report and dashboard."""
    out = (df.groupby("category").size().rename("flows").reset_index()
             .sort_values("flows", ascending=False))
    out["share"] = out["flows"] / out["flows"].sum()
    return out.reset_index(drop=True)


def family_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby(["category", "label"]).size().rename("flows")
             .reset_index().sort_values(["category", "flows"],
                                        ascending=[True, False]))
    return out.reset_index(drop=True)

