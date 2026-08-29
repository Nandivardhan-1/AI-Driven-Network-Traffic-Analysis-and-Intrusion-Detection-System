"""Central configuration: paths, dataset schema, attack taxonomy, feature groups.

Everything that another module might want to hard-code lives here instead, so the
report, the CLI and the tests all agree on one definition of the data.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG_DIR)

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "figures")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
ALERT_DIR = os.path.join(ARTIFACT_DIR, "alerts")

for _d in (RAW_DIR, FIGURE_DIR, MODEL_DIR, ALERT_DIR):
    os.makedirs(_d, exist_ok=True)

# Canonical NSL-KDD filenames expected inside data/raw/
NSL_TRAIN_FILE = "KDDTrain+.txt"
NSL_TEST_FILE = "KDDTest+.txt"

RANDOM_SEED = 20260826

# --------------------------------------------------------------------------- #
# NSL-KDD schema (41 features + label + difficulty), in file order
# --------------------------------------------------------------------------- #
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
LABEL_COLUMN = "label"

# --------------------------------------------------------------------------- #
# Attack taxonomy: raw NSL-KDD label -> coarse category
# --------------------------------------------------------------------------- #
ATTACK_CATEGORY = {
    # Denial of Service / flooding
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Reconnaissance / scanning
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # Remote-to-local: credential brute force, guessing, illicit remote access
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    "worm": "r2l",
    # User-to-root privilege escalation
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]

CATEGORY_DESCRIPTION = {
    "normal": "Benign background traffic",
    "dos": "Denial of service / flooding",
    "probe": "Reconnaissance / port scanning",
    "r2l": "Remote-to-local (brute force, credential guessing)",
    "u2r": "User-to-root privilege escalation",
}

# Attacks that appear only in KDDTest+ and never in KDDTrain+. Used to quantify
# zero-day generalisation, which is the basis of demonstration Scenario 5.
NOVEL_TEST_ATTACKS = [
    "apache2", "httptunnel", "mailbomb", "mscan", "named", "processtable",
    "ps", "saint", "sendmail", "snmpgetattack", "snmpguess", "sqlattack",
    "udpstorm", "worm", "xlock", "xsnoop", "xterm",
]


def category_of(label: str) -> str:
    """Map a raw NSL-KDD label to one of CATEGORIES."""
    label = str(label).strip().rstrip(".").lower()
    if label in ("normal", "benign"):
        return "normal"
    return ATTACK_CATEGORY.get(label, "unknown")


# --------------------------------------------------------------------------- #
# Feature groups. Section B of the brief asks which features were selected and
# why; these groupings are what the report and the dashboard cite.
# --------------------------------------------------------------------------- #
FEATURE_GROUPS = {
    "volume": {
        "why": "Byte and packet volumes separate bulk transfers and amplification "
               "floods from interactive sessions.",
        "features": ["src_bytes", "dst_bytes", "duration", "byte_ratio",
                     "bytes_per_second", "log_src_bytes", "log_dst_bytes"],
    },
    "connection_rate": {
        "why": "Counts of connections to the same host/service in the trailing "
               "2-second window are the primary signal for scanning and flooding.",
        "features": ["count", "srv_count", "dst_host_count", "dst_host_srv_count",
                     "conn_per_second"],
    },
    "error_behaviour": {
        "why": "SYN-error and rejection rates spike when a scanner touches closed "
               "ports or a SYN flood exhausts the backlog.",
        "features": ["serror_rate", "srv_serror_rate", "rerror_rate",
                     "srv_rerror_rate", "dst_host_serror_rate",
                     "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                     "dst_host_srv_rerror_rate", "error_pressure"],
    },
    "service_spread": {
        "why": "A host contacted across many different services with little repeat "
               "is horizontal scanning; a single service repeated is a flood.",
        "features": ["same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                     "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "service_entropy_proxy"],
    },
    "authentication": {
        "why": "Failed logins, guest logins and shell/root activity are the only "
               "content-level evidence of brute force and privilege escalation.",
        "features": ["num_failed_logins", "logged_in", "is_guest_login",
                     "is_host_login", "root_shell", "su_attempted", "num_root",
                     "hot", "num_compromised", "num_file_creations",
                     "num_shells", "num_access_files", "auth_pressure"],
    },
    "protocol": {
        "why": "Protocol, service and TCP flag state give the categorical context "
               "that makes the numeric features interpretable.",
        "features": ["protocol_type", "service", "flag", "land",
                     "wrong_fragment", "urgent", "num_outbound_cmds"],
    },
}

# Signature-layer thresholds. Kept here so the report can quote them and the
# tests can assert against them without duplicating magic numbers.
RULE_THRESHOLDS = {
    "syn_flood_serror_rate": 0.85,
    "syn_flood_min_count": 100,
    "portscan_diff_srv_rate": 0.55,
    "portscan_min_dst_host_count": 30,
    "portscan_rerror_rate": 0.60,
    "hostsweep_srv_diff_host_rate": 0.70,
    "bruteforce_min_failed_logins": 2,
    "icmp_flood_min_src_bytes": 800,
    "malformed_min_wrong_fragment": 1,
}

# Hybrid fusion weights (see detect/hybrid.py).
FUSION = {
    "anomaly_alert_percentile": 99.0,   # contamination boundary fitted on benign
    "supervised_alert_threshold": 0.50,
    "severity_bands": [(0.85, "critical"), (0.65, "high"), (0.45, "medium")],
}

# Severity vocabulary, lowest to highest. Used for sorting the alert queue and for
# the ``--min-severity`` filter in the CLI.
SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]
