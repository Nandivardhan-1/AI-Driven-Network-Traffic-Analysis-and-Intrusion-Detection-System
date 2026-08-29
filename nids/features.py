"""Feature extraction and encoding.

Three kinds of feature reach the models:

1. **Native connection features** -- the numeric NSL-KDD columns, used as-is.
2. **Derived ratios and pressures** -- normalisations that make a threshold
   meaningful across different traffic volumes (bytes per second rather than raw
   bytes, error pressure rather than four separate error rates).
3. **Encoded categoricals** -- protocol/service/flag, encoded three ways because
   each carries different information: an ordinal code for the trees, a training
   frequency so "a service nobody uses" is expressible as a number, and boolean
   indicators for the connection states that matter operationally.

``FeatureSpace`` is fitted on training data only. Everything it learns (category
vocabularies, service frequencies, scaler statistics) comes from the training
split, so evaluation on the test split stays honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLUMNS, FEATURE_GROUPS, NSL_COLUMNS
from .ml.preprocess import OrdinalEncoder, StandardScaler

# Native numeric NSL-KDD columns, in schema order.
NATIVE_NUMERIC = [c for c in NSL_COLUMNS
                  if c not in CATEGORICAL_COLUMNS + ["label", "difficulty"]]

SERROR_FLAGS = ("s0", "s1", "s2", "s3")
RERROR_FLAGS = ("rej", "rsto", "rstr", "rstos0", "oth")

DERIVED_DOC = {
    "log_src_bytes": "log1p(src_bytes) -- byte counts span six orders of "
                     "magnitude; the log makes the scale learnable.",
    "log_dst_bytes": "log1p(dst_bytes).",
    "byte_ratio": "src_bytes / (src_bytes + dst_bytes + 1). Near 1 means the "
                  "client talked and the server never answered, the shape of a "
                  "scan or a flood; near 0 is a download.",
    "bytes_per_second": "(src_bytes + dst_bytes) / (duration + 1). Separates a "
                        "long slow session from a burst of the same volume.",
    "conn_per_second": "count / (duration + 1). Connection attempt rate to the "
                       "same host, the primary flood signal.",
    "error_pressure": "max(serror_rate, srv_serror_rate) + max(rerror_rate, "
                      "srv_rerror_rate), in [0, 2]. One number for 'connections "
                      "here are failing', which is what both scanning and SYN "
                      "flooding produce.",
    "service_entropy_proxy": "diff_srv_rate * dst_host_diff_srv_rate. High only "
                             "when a source sprays many services both in the "
                             "2-second window and across the host's history, "
                             "which is horizontal scanning rather than a busy "
                             "server.",
    "auth_pressure": "num_failed_logins + 2*root_shell + 2*su_attempted + "
                     "hot/5 + is_guest_login. The only feature block that can "
                     "see brute force and privilege escalation, because those "
                     "attacks are low-volume by design.",
    "service_freq": "Share of training connections using this service. Rare or "
                    "never-before-seen services score near zero.",
    "flag_serror": "1 if the TCP state is a half-open/incomplete handshake "
                   "(S0/S1/S2/S3).",
    "flag_rerror": "1 if the connection was rejected or reset (REJ/RSTO/RSTR).",
    "proto_tcp": "Protocol indicator.",
    "proto_udp": "Protocol indicator.",
    "proto_icmp": "Protocol indicator.",
}

DERIVED_NUMERIC = ["log_src_bytes", "log_dst_bytes", "byte_ratio",
                   "bytes_per_second", "conn_per_second", "error_pressure",
                   "service_entropy_proxy", "auth_pressure", "service_freq",
                   "flag_serror", "flag_rerror", "proto_tcp", "proto_udp",
                   "proto_icmp"]

ENCODED_CATEGORICAL = ["protocol_type_code", "service_code", "flag_code"]


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Numeric column accessor that tolerates a missing column (pcap path)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(float)
    return np.zeros(len(df))


def _text(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].astype(str).str.strip().str.lower().to_numpy(dtype=object)
    return np.full(len(df), "unknown", dtype=object)


def _nonneg(a: np.ndarray) -> np.ndarray:
    """Clean a physical quantity that cannot be negative or infinite.

    Byte counts, durations and connection counts are all non-negative by
    definition, so a negative or infinite value is corrupt input rather than a
    measurement. Sanitising here keeps the divisions below finite: a corrupt
    ``duration`` of -1 would otherwise make ``count / (duration + 1)`` a 0/0.
    """
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def derive(df: pd.DataFrame, service_freq: dict[str, float] | None = None) -> pd.DataFrame:
    """Compute the derived feature block. Pure function, no fitted state except
    the optional training service-frequency table."""
    src, dst = _nonneg(_col(df, "src_bytes")), _nonneg(_col(df, "dst_bytes"))
    dur, cnt = _nonneg(_col(df, "duration")), _nonneg(_col(df, "count"))
    proto, flag, svc = _text(df, "protocol_type"), _text(df, "flag"), _text(df, "service")
    freq_map = service_freq or {}

    out = pd.DataFrame(index=df.index)
    out["log_src_bytes"] = np.log1p(np.clip(src, 0, None))
    out["log_dst_bytes"] = np.log1p(np.clip(dst, 0, None))
    out["byte_ratio"] = src / (src + dst + 1.0)
    out["bytes_per_second"] = (src + dst) / (dur + 1.0)
    out["conn_per_second"] = cnt / (dur + 1.0)
    out["error_pressure"] = (
        np.maximum(_col(df, "serror_rate"), _col(df, "srv_serror_rate"))
        + np.maximum(_col(df, "rerror_rate"), _col(df, "srv_rerror_rate"))
    )
    out["service_entropy_proxy"] = (_col(df, "diff_srv_rate")
                                    * _col(df, "dst_host_diff_srv_rate"))
    out["auth_pressure"] = (
        _col(df, "num_failed_logins")
        + 2.0 * _col(df, "root_shell")
        + 2.0 * _col(df, "su_attempted")
        + _col(df, "hot") / 5.0
        + _col(df, "is_guest_login")
    )
    out["service_freq"] = np.array([freq_map.get(s, 0.0) for s in svc])
    out["flag_serror"] = np.isin(flag, SERROR_FLAGS).astype(float)
    out["flag_rerror"] = np.isin(flag, RERROR_FLAGS).astype(float)
    out["proto_tcp"] = (proto == "tcp").astype(float)
    out["proto_udp"] = (proto == "udp").astype(float)
    out["proto_icmp"] = (proto == "icmp").astype(float)
    return out


class FeatureSpace:
    """Fitted feature pipeline: raw connection frame -> float design matrix."""

    def __init__(self, numeric_columns: list[str] | None = None) -> None:
        self.numeric_columns = numeric_columns or list(NATIVE_NUMERIC)
        self.encoders: dict[str, OrdinalEncoder] = {}
        self.service_freq: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "FeatureSpace":
        svc = _text(df, "service")
        vals, counts = np.unique(svc, return_counts=True)
        self.service_freq = dict(zip(vals.tolist(), (counts / counts.sum()).tolist()))
        for col in CATEGORICAL_COLUMNS:
            self.encoders[col] = OrdinalEncoder().fit(_text(df, col))
        self.feature_names = (list(self.numeric_columns) + list(DERIVED_NUMERIC)
                              + list(ENCODED_CATEGORICAL))
        self._fitted = True
        self.scaler.fit(self._matrix(df))
        return self

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        native = np.column_stack([_col(df, c) for c in self.numeric_columns])
        derived = derive(df, self.service_freq)[DERIVED_NUMERIC].to_numpy(float)
        codes = np.column_stack([self.encoders[c].transform(_text(df, c)).astype(float)
                                 for c in CATEGORICAL_COLUMNS])
        X = np.hstack([native, derived, codes])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureSpace must be fitted first")
        return self._matrix(df)

    def transform_scaled(self, df: pd.DataFrame) -> np.ndarray:
        """Standardised matrix. Trees ignore scale; the isolation forest does not,
        because its split points are drawn uniformly across each feature's range."""
        return self.scaler.transform(self.transform(df))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    def index_of(self, name: str) -> int:
        return self.feature_names.index(name)

    def group_of(self, name: str) -> str:
        """Which documented feature group a column belongs to (for the UI)."""
        base = name.replace("_code", "")
        for group, spec in FEATURE_GROUPS.items():
            if name in spec["features"] or base in spec["features"]:
                return group
        return "derived" if name in DERIVED_NUMERIC else "other"

    def describe(self, name: str) -> str:
        if name in DERIVED_DOC:
            return DERIVED_DOC[name]
        group = self.group_of(name)
        if group in FEATURE_GROUPS:
            return FEATURE_GROUPS[group]["why"]
        return "Native NSL-KDD connection feature."

    @property
    def n_features(self) -> int:
        return len(self.feature_names)



