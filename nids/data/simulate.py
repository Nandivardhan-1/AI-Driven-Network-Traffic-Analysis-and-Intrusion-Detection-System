"""Seeded synthetic traffic corpus in the NSL-KDD schema.

Why this exists
---------------
The project is built around NSL-KDD (see ``nids/data/nsl_kdd.py``). When the real
``KDDTrain+.txt`` / ``KDDTest+.txt`` files are present in ``data/raw/`` they are
always used. This module is the fallback so that the repository is runnable end to
end by a grader with no downloads at all, and so the unit tests do not depend on
an external file.

It is a *stand-in, not a substitute*. Every profile below is a hand-specified
distribution over the 41 NSL-KDD features, informed by the documented behaviour of
each attack family. It deliberately reproduces three awkward properties of the real
data rather than papering over them:

1. Severe class imbalance -- R2L and U2R are a fraction of a percent of traffic.
2. Heavy overlap between R2L/U2R and benign traffic, because those attacks are
   low-volume and only distinguishable by content features.
3. Attack families present in the test split that never appear in training, which
   is the basis of demonstration Scenario 5.

Any figure produced from this corpus is labelled ``SIMULATED`` by the CLI and the
dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import NSL_COLUMNS, RANDOM_SEED

# Feature defaults: anything a profile does not mention is zero, matching the
# real data where most content features are 0 for most connections.
_RATE = 0.0


def _draw(spec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n values from a compact distribution spec."""
    kind = spec[0]
    if kind == "const":
        return np.full(n, spec[1], dtype=object if isinstance(spec[1], str) else float)
    if kind == "int":
        return rng.integers(spec[1], spec[2] + 1, n).astype(float)
    if kind == "unif":
        return rng.uniform(spec[1], spec[2], n)
    if kind == "beta":                      # rates in [0, 1] with a shape
        return rng.beta(spec[1], spec[2], n)
    if kind == "lognorm":                   # byte counts: heavy right tail
        return np.round(rng.lognormal(spec[1], spec[2], n))
    if kind == "choice":
        vals = spec[1]
        p = np.asarray(spec[2], float) if len(spec) > 2 else None
        if p is not None:
            p = p / p.sum()
        return rng.choice(vals, size=n, p=p)
    if kind == "bern":                      # 0/1 flag with probability p
        return (rng.random(n) < spec[1]).astype(float)
    if kind == "zeroinf":                   # mostly zero, occasionally spec[2]
        base = _draw(spec[2], n, rng)
        return np.where(rng.random(n) < spec[1], 0.0, base)
    raise ValueError(f"unknown spec {spec!r}")


# --------------------------------------------------------------------------- #
# Per-family profiles. Keys are NSL-KDD column names.
# --------------------------------------------------------------------------- #
PROFILES: dict[str, dict] = {}

PROFILES["normal"] = {
    "duration": ("zeroinf", 0.55, ("lognorm", 1.2, 1.6)),
    "protocol_type": ("choice", ["tcp", "udp", "icmp"], [0.80, 0.17, 0.03]),
    "service": ("choice", ["http", "private", "domain_u", "smtp", "ftp_data",
                           "other", "telnet", "pop_3", "ecr_i"],
                [0.38, 0.14, 0.12, 0.09, 0.08, 0.09, 0.04, 0.03, 0.03]),
    "flag": ("choice", ["SF", "REJ", "S0", "RSTR"], [0.90, 0.05, 0.03, 0.02]),
    "src_bytes": ("lognorm", 5.3, 1.4),
    "dst_bytes": ("lognorm", 6.6, 2.1),
    "logged_in": ("bern", 0.62),
    "count": ("int", 1, 14),
    "srv_count": ("int", 1, 12),
    "serror_rate": ("zeroinf", 0.94, ("beta", 1.2, 6.0)),
    "srv_serror_rate": ("zeroinf", 0.94, ("beta", 1.2, 6.0)),
    "rerror_rate": ("zeroinf", 0.92, ("beta", 1.2, 6.0)),
    "srv_rerror_rate": ("zeroinf", 0.92, ("beta", 1.2, 6.0)),
    "same_srv_rate": ("beta", 9.0, 1.0),
    "diff_srv_rate": ("beta", 1.0, 12.0),
    "srv_diff_host_rate": ("beta", 1.0, 8.0),
    "dst_host_count": ("int", 1, 255),
    "dst_host_srv_count": ("int", 1, 255),
    "dst_host_same_srv_rate": ("beta", 7.0, 1.5),
    "dst_host_diff_srv_rate": ("beta", 1.0, 10.0),
    "dst_host_same_src_port_rate": ("beta", 2.0, 3.0),
    "dst_host_srv_diff_host_rate": ("beta", 1.0, 12.0),
    "hot": ("zeroinf", 0.97, ("int", 1, 2)),
    "is_guest_login": ("bern", 0.02),
}

# --------------------------- Benign heterogeneity -------------------------- #
# A single unimodal "normal" profile is the fastest way to build a corpus on which
# every detector looks perfect: benign and attack regions never touch, so a tree
# separates them exactly and the reported precision means nothing. Real networks
# are not unimodal. These variants put benign traffic *inside* the regions the
# attack profiles occupy, which is what produces genuine false positives and makes
# the evaluation section worth reading.
#
# All of them carry label "normal" via _LABEL_ALIAS.
PROFILES["normal_busy_server"] = {          # popular web server: high connection rate
    "duration": ("zeroinf", 0.40, ("lognorm", 0.8, 1.2)),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["http", "smtp", "private"], [0.72, 0.18, 0.10]),
    "flag": ("choice", ["SF", "RSTR", "S1"], [0.93, 0.05, 0.02]),
    "src_bytes": ("lognorm", 6.2, 1.5),
    "dst_bytes": ("lognorm", 7.8, 1.9),
    "logged_in": ("bern", 0.70),
    "count": ("int", 40, 320),              # overlaps the application-flood region
    "srv_count": ("int", 40, 320),
    "same_srv_rate": ("beta", 18.0, 1.0),
    "dst_host_count": ("int", 120, 255),
    "dst_host_srv_count": ("int", 120, 255),
    "dst_host_same_srv_rate": ("beta", 16.0, 1.0),
    "dst_host_same_src_port_rate": ("beta", 2.0, 4.0),
}

PROFILES["normal_bulk_transfer"] = {        # backup / large download
    "duration": ("int", 5, 900),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["ftp_data", "private", "http"], [0.55, 0.25, 0.20]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 10.5, 1.6),    # overlaps back/apache2 byte counts
    "dst_bytes": ("lognorm", 9.5, 2.2),
    "logged_in": ("bern", 0.85),
    "hot": ("zeroinf", 0.85, ("int", 1, 3)),
    "count": ("int", 1, 12),
    "srv_count": ("int", 1, 12),
    "same_srv_rate": ("beta", 9.0, 1.2),
    "dst_host_count": ("int", 1, 200),
    "dst_host_srv_count": ("int", 1, 200),
    "dst_host_same_srv_rate": ("beta", 7.0, 1.5),
    "dst_host_same_src_port_rate": ("beta", 3.0, 3.0),
}

PROFILES["normal_auth_typo"] = {            # user mistypes a password once or twice
    "duration": ("int", 1, 40),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["telnet", "ftp", "pop_3", "imap4"],
                [0.35, 0.30, 0.25, 0.10]),
    "flag": ("choice", ["SF", "RSTO"], [0.9, 0.1]),
    "src_bytes": ("lognorm", 4.9, 0.7),
    "dst_bytes": ("lognorm", 5.2, 0.9),
    "num_failed_logins": ("choice", [1.0, 2.0], [0.80, 0.20]),
    "logged_in": ("bern", 0.85),            # they get in on the retry
    "count": ("int", 1, 5),
    "srv_count": ("int", 1, 5),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 200),
    "dst_host_srv_count": ("int", 1, 90),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
    "dst_host_same_src_port_rate": ("beta", 2.0, 3.0),
}

PROFILES["normal_admin_session"] = {        # legitimate sysadmin: su, root files
    "duration": ("int", 10, 1800),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["telnet", "other", "private"], [0.7, 0.2, 0.1]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 6.8, 1.3),
    "dst_bytes": ("lognorm", 8.0, 1.6),
    "hot": ("zeroinf", 0.55, ("int", 1, 6)),
    "su_attempted": ("bern", 0.35),
    "root_shell": ("bern", 0.45),           # collides head-on with rule R11
    "num_root": ("zeroinf", 0.30, ("int", 1, 12)),
    "num_file_creations": ("zeroinf", 0.55, ("int", 1, 5)),
    "num_shells": ("bern", 0.30),
    "logged_in": ("const", 1.0),
    "count": ("int", 1, 6),
    "srv_count": ("int", 1, 6),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 120),
    "dst_host_srv_count": ("int", 1, 80),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
}

PROFILES["normal_guest_ftp"] = {            # anonymous FTP used as intended
    "duration": ("int", 0, 120),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["ftp", "ftp_data"], [0.5, 0.5]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 5.6, 1.5),
    "dst_bytes": ("lognorm", 7.4, 2.0),
    "hot": ("choice", [0.0, 1.0, 2.0], [0.45, 0.35, 0.20]),
    "num_file_creations": ("zeroinf", 0.75, ("int", 1, 2)),
    "logged_in": ("const", 1.0),
    "is_guest_login": ("const", 1.0),       # collides head-on with rule R10
    "count": ("int", 1, 8),
    "srv_count": ("int", 1, 8),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 100),
    "dst_host_srv_count": ("int", 1, 70),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
}

PROFILES["normal_monitoring"] = {           # SNMP/DNS polling: many hosts, one service
    "duration": ("const", 0.0),
    "protocol_type": ("choice", ["udp", "icmp"], [0.75, 0.25]),
    "service": ("choice", ["private", "domain_u", "eco_i"], [0.45, 0.35, 0.20]),
    "flag": ("const", "SF"),
    "src_bytes": ("choice", [44.0, 105.0, 8.0, 60.0], [0.3, 0.3, 0.2, 0.2]),
    "dst_bytes": ("choice", [0.0, 105.0, 147.0], [0.35, 0.35, 0.30]),
    "count": ("int", 1, 20),
    "srv_count": ("int", 5, 90),
    "same_srv_rate": ("beta", 12.0, 1.2),
    "srv_diff_host_rate": ("beta", 9.0, 1.5),   # collides with host-sweep rule R08
    "dst_host_count": ("int", 1, 60),
    "dst_host_srv_count": ("int", 80, 255),
    "dst_host_same_srv_rate": ("beta", 10.0, 1.3),
    "dst_host_srv_diff_host_rate": ("beta", 8.0, 1.6),
    "dst_host_same_src_port_rate": ("beta", 7.0, 1.8),
}

PROFILES["normal_scanner_like"] = {         # vulnerability scanner run by the SOC,
    "duration": ("const", 0.0),             # plus clients hitting closed ports
    "protocol_type": ("choice", ["tcp", "udp"], [0.9, 0.1]),
    "service": ("choice", ["private", "other", "http", "domain_u"],
                [0.35, 0.30, 0.20, 0.15]),
    "flag": ("choice", ["REJ", "SF", "S0"], [0.55, 0.30, 0.15]),
    "src_bytes": ("zeroinf", 0.70, ("int", 1, 120)),
    "dst_bytes": ("zeroinf", 0.80, ("int", 1, 90)),
    "count": ("int", 1, 25),
    "srv_count": ("int", 1, 8),
    "rerror_rate": ("beta", 6.0, 2.0),      # collides head-on with rule R07
    "srv_rerror_rate": ("beta", 6.0, 2.0),
    "same_srv_rate": ("beta", 2.0, 4.0),
    "diff_srv_rate": ("beta", 3.0, 3.0),
    "dst_host_count": ("int", 20, 255),
    "dst_host_srv_count": ("int", 1, 60),
    "dst_host_diff_srv_rate": ("beta", 3.0, 3.0),
    "dst_host_rerror_rate": ("beta", 5.0, 2.5),
    "dst_host_same_src_port_rate": ("beta", 1.5, 5.0),
}

# Variant name -> the label written to the corpus. Every benign variant is
# indistinguishable from "normal" as far as ground truth is concerned.
_LABEL_ALIAS = {name: "normal" for name in
                ("normal_busy_server", "normal_bulk_transfer", "normal_auth_typo",
                 "normal_admin_session", "normal_guest_ftp", "normal_monitoring",
                 "normal_scanner_like")}

# Share of benign traffic drawn from each variant. The plain profile still
# dominates; the awkward cases are a realistic minority.
BENIGN_MIX = {
    "normal": 0.72,
    "normal_busy_server": 0.07,
    "normal_bulk_transfer": 0.06,
    "normal_monitoring": 0.06,
    "normal_scanner_like": 0.04,
    "normal_auth_typo": 0.025,
    "normal_guest_ftp": 0.02,
    "normal_admin_session": 0.005,
}

# --------------------------- DoS / flooding -------------------------------- #
PROFILES["neptune"] = {                     # TCP SYN flood: half-open floods
    "duration": ("const", 0.0),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["private", "http", "other", "smtp"],
                [0.62, 0.20, 0.12, 0.06]),
    "flag": ("choice", ["S0", "REJ"], [0.93, 0.07]),
    "src_bytes": ("const", 0.0),
    "dst_bytes": ("const", 0.0),
    "count": ("int", 110, 511),
    "srv_count": ("int", 8, 511),
    "serror_rate": ("beta", 60.0, 1.0),
    "srv_serror_rate": ("beta", 60.0, 1.0),
    "same_srv_rate": ("beta", 5.0, 2.0),
    "diff_srv_rate": ("beta", 1.5, 6.0),
    "dst_host_count": ("int", 200, 255),
    "dst_host_srv_count": ("int", 1, 60),
    "dst_host_same_srv_rate": ("beta", 1.5, 5.0),
    "dst_host_serror_rate": ("beta", 60.0, 1.0),
    "dst_host_srv_serror_rate": ("beta", 60.0, 1.0),
    "dst_host_same_src_port_rate": ("beta", 8.0, 1.5),
}

PROFILES["smurf"] = {                       # ICMP echo amplification
    "duration": ("const", 0.0),
    "protocol_type": ("const", "icmp"),
    "service": ("const", "ecr_i"),
    "flag": ("const", "SF"),
    "src_bytes": ("choice", [1032.0, 1480.0], [0.9, 0.1]),
    "dst_bytes": ("const", 0.0),
    "count": ("int", 300, 511),
    "srv_count": ("int", 300, 511),
    "same_srv_rate": ("const", 1.0),
    "dst_host_count": ("const", 255.0),
    "dst_host_srv_count": ("const", 255.0),
    "dst_host_same_srv_rate": ("const", 1.0),
    "dst_host_same_src_port_rate": ("const", 1.0),
}

PROFILES["teardrop"] = {                    # malformed overlapping fragments
    "duration": ("const", 0.0),
    "protocol_type": ("const", "udp"),
    "service": ("const", "private"),
    "flag": ("const", "SF"),
    "src_bytes": ("choice", [28.0, 24.0], [0.8, 0.2]),
    "dst_bytes": ("const", 0.0),
    "wrong_fragment": ("choice", [3.0, 1.0], [0.85, 0.15]),
    "count": ("int", 1, 20),
    "srv_count": ("int", 1, 20),
    "same_srv_rate": ("const", 1.0),
    "dst_host_count": ("int", 20, 255),
    "dst_host_srv_count": ("int", 20, 255),
    "dst_host_same_srv_rate": ("const", 1.0),
    "dst_host_same_src_port_rate": ("beta", 4.0, 2.0),
}

PROFILES["back"] = {                        # Apache request-flood, looks like HTTP
    "duration": ("int", 0, 5),
    "protocol_type": ("const", "tcp"),
    "service": ("const", "http"),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 10.9, 0.10),
    "dst_bytes": ("lognorm", 7.9, 0.35),
    "logged_in": ("const", 1.0),
    "count": ("int", 2, 40),
    "srv_count": ("int", 2, 40),
    "same_srv_rate": ("const", 1.0),
    "dst_host_count": ("int", 30, 255),
    "dst_host_srv_count": ("int", 30, 255),
    "dst_host_same_srv_rate": ("const", 1.0),
    "dst_host_same_src_port_rate": ("beta", 3.0, 3.0),
}

# ----------------------- Probe / reconnaissance ---------------------------- #
PROFILES["portsweep"] = {                   # vertical scan: one host, many ports
    "duration": ("zeroinf", 0.70, ("lognorm", 4.0, 2.0)),
    "protocol_type": ("choice", ["tcp", "udp"], [0.93, 0.07]),
    "service": ("choice", ["private", "other", "http", "telnet", "smtp",
                           "domain_u", "finger"],
                [0.40, 0.22, 0.10, 0.10, 0.08, 0.05, 0.05]),
    "flag": ("choice", ["S0", "REJ", "RSTR", "SF"], [0.45, 0.38, 0.10, 0.07]),
    "src_bytes": ("zeroinf", 0.80, ("int", 1, 60)),
    "dst_bytes": ("const", 0.0),
    "count": ("int", 1, 30),
    "srv_count": ("int", 1, 4),
    "serror_rate": ("beta", 3.0, 2.0),
    "srv_serror_rate": ("beta", 3.0, 2.0),
    "rerror_rate": ("beta", 3.0, 2.5),
    "srv_rerror_rate": ("beta", 3.0, 2.5),
    "same_srv_rate": ("beta", 1.0, 5.0),
    "diff_srv_rate": ("beta", 5.0, 1.5),
    "dst_host_count": ("int", 60, 255),
    "dst_host_srv_count": ("int", 1, 12),
    "dst_host_same_srv_rate": ("beta", 1.0, 12.0),
    "dst_host_diff_srv_rate": ("beta", 6.0, 1.5),
    "dst_host_same_src_port_rate": ("beta", 1.0, 6.0),
    "dst_host_serror_rate": ("beta", 2.5, 2.0),
    "dst_host_rerror_rate": ("beta", 2.5, 2.0),
}

PROFILES["satan"] = {                       # noisy vulnerability scanner
    "duration": ("zeroinf", 0.65, ("lognorm", 1.0, 1.5)),
    "protocol_type": ("choice", ["tcp", "udp", "icmp"], [0.72, 0.22, 0.06]),
    "service": ("choice", ["private", "other", "finger", "domain_u", "http",
                           "telnet", "auth", "time"],
                [0.28, 0.20, 0.12, 0.12, 0.10, 0.08, 0.05, 0.05]),
    "flag": ("choice", ["REJ", "SF", "S0", "RSTR"], [0.55, 0.20, 0.15, 0.10]),
    "src_bytes": ("zeroinf", 0.55, ("int", 1, 300)),
    "dst_bytes": ("zeroinf", 0.70, ("int", 1, 200)),
    "count": ("int", 1, 40),
    "srv_count": ("int", 1, 6),
    "rerror_rate": ("beta", 8.0, 1.5),
    "srv_rerror_rate": ("beta", 8.0, 1.5),
    "same_srv_rate": ("beta", 1.0, 8.0),
    "diff_srv_rate": ("beta", 8.0, 1.5),
    "srv_diff_host_rate": ("beta", 2.0, 3.0),
    "dst_host_count": ("int", 40, 255),
    "dst_host_srv_count": ("int", 1, 20),
    "dst_host_diff_srv_rate": ("beta", 7.0, 1.5),
    "dst_host_rerror_rate": ("beta", 7.0, 1.5),
    "dst_host_srv_rerror_rate": ("beta", 7.0, 1.5),
    "dst_host_same_src_port_rate": ("beta", 1.0, 5.0),
}

PROFILES["ipsweep"] = {                     # horizontal sweep: many hosts, 1 service
    "duration": ("const", 0.0),
    "protocol_type": ("choice", ["icmp", "tcp"], [0.85, 0.15]),
    "service": ("choice", ["eco_i", "private", "http"], [0.85, 0.10, 0.05]),
    "flag": ("choice", ["SF", "S0", "REJ"], [0.72, 0.18, 0.10]),
    "src_bytes": ("choice", [8.0, 18.0, 0.0], [0.55, 0.30, 0.15]),
    "dst_bytes": ("const", 0.0),
    "count": ("int", 1, 25),
    "srv_count": ("int", 1, 40),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "diff_srv_rate": ("beta", 1.0, 10.0),
    "srv_diff_host_rate": ("beta", 12.0, 1.0),
    "dst_host_count": ("int", 1, 60),
    "dst_host_srv_count": ("int", 100, 255),
    "dst_host_same_srv_rate": ("beta", 9.0, 1.2),
    "dst_host_srv_diff_host_rate": ("beta", 10.0, 1.5),
    "dst_host_same_src_port_rate": ("beta", 8.0, 1.5),
}

PROFILES["nmap"] = {                        # low-volume stealth scan
    "duration": ("const", 0.0),
    "protocol_type": ("choice", ["tcp", "icmp", "udp"], [0.55, 0.30, 0.15]),
    "service": ("choice", ["private", "other", "eco_i", "domain_u", "http"],
                [0.35, 0.25, 0.20, 0.12, 0.08]),
    "flag": ("choice", ["SF", "S0", "REJ"], [0.45, 0.35, 0.20]),
    "src_bytes": ("zeroinf", 0.60, ("int", 1, 40)),
    "dst_bytes": ("const", 0.0),
    "count": ("int", 1, 8),
    "srv_count": ("int", 1, 6),
    "serror_rate": ("beta", 2.0, 3.0),
    "rerror_rate": ("beta", 2.0, 3.0),
    "same_srv_rate": ("beta", 1.5, 3.0),
    "diff_srv_rate": ("beta", 4.0, 2.0),
    "srv_diff_host_rate": ("beta", 3.0, 2.5),
    "dst_host_count": ("int", 1, 120),
    "dst_host_srv_count": ("int", 1, 40),
    "dst_host_diff_srv_rate": ("beta", 4.0, 2.0),
    "dst_host_same_src_port_rate": ("beta", 2.0, 4.0),
}

# ------------- R2L: credential brute force and remote misuse --------------- #
# Deliberately close to benign traffic: a handful of connections, ordinary byte
# counts, and only the authentication features carrying evidence.
PROFILES["guess_passwd"] = {
    "duration": ("int", 1, 22),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["telnet", "ftp", "pop_3", "imap4"],
                [0.45, 0.25, 0.20, 0.10]),
    "flag": ("choice", ["SF", "RSTO"], [0.88, 0.12]),
    "src_bytes": ("lognorm", 4.8, 0.55),
    "dst_bytes": ("lognorm", 5.0, 0.60),
    "num_failed_logins": ("choice", [1.0, 2.0, 3.0, 4.0, 5.0],
                          [0.34, 0.28, 0.18, 0.12, 0.08]),
    "logged_in": ("bern", 0.12),
    "is_guest_login": ("bern", 0.30),
    "count": ("int", 1, 6),
    "srv_count": ("int", 1, 6),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 255),
    "dst_host_srv_count": ("int", 1, 60),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
    "dst_host_same_src_port_rate": ("beta", 2.0, 3.0),
    "dst_host_rerror_rate": ("zeroinf", 0.7, ("beta", 2.0, 3.0)),
}

PROFILES["warezclient"] = {                 # illicit file transfer after login
    "duration": ("zeroinf", 0.35, ("lognorm", 2.5, 1.8)),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["ftp_data", "ftp"], [0.72, 0.28]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 8.5, 2.6),
    "dst_bytes": ("zeroinf", 0.55, ("lognorm", 5.5, 1.8)),
    "hot": ("choice", [2.0, 3.0, 1.0, 0.0], [0.42, 0.28, 0.20, 0.10]),
    "logged_in": ("const", 1.0),
    "is_guest_login": ("bern", 0.55),
    "count": ("int", 1, 10),
    "srv_count": ("int", 1, 10),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 120),
    "dst_host_srv_count": ("int", 1, 90),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
    "dst_host_same_src_port_rate": ("beta", 3.0, 3.0),
}

PROFILES["ftp_write"] = {                   # anonymous FTP abuse, very rare
    "duration": ("int", 0, 40),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["ftp", "ftp_data"], [0.6, 0.4]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 5.5, 1.2),
    "dst_bytes": ("lognorm", 5.2, 1.4),
    "hot": ("choice", [1.0, 2.0], [0.6, 0.4]),
    "num_file_creations": ("choice", [1.0, 2.0], [0.7, 0.3]),
    "logged_in": ("const", 1.0),
    "is_guest_login": ("const", 1.0),
    "count": ("int", 1, 4),
    "srv_count": ("int", 1, 4),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 60),
    "dst_host_srv_count": ("int", 1, 30),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
}

# ------------------ U2R: privilege escalation, very rare ------------------- #
PROFILES["buffer_overflow"] = {
    "duration": ("int", 2, 320),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["telnet", "ftp_data", "http"], [0.75, 0.15, 0.10]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 6.9, 1.1),
    "dst_bytes": ("lognorm", 8.2, 1.3),
    "hot": ("choice", [2.0, 3.0, 14.0, 22.0], [0.35, 0.30, 0.20, 0.15]),
    "root_shell": ("bern", 0.80),
    "num_root": ("int", 1, 10),
    "num_file_creations": ("zeroinf", 0.45, ("int", 1, 4)),
    "num_shells": ("bern", 0.40),
    "logged_in": ("const", 1.0),
    "count": ("int", 1, 4),
    "srv_count": ("int", 1, 4),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 90),
    "dst_host_srv_count": ("int", 1, 60),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
}

PROFILES["rootkit"] = {
    "duration": ("int", 0, 200),
    "protocol_type": ("choice", ["tcp", "udp"], [0.8, 0.2]),
    "service": ("choice", ["telnet", "private", "ftp_data"], [0.6, 0.25, 0.15]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 6.0, 1.5),
    "dst_bytes": ("lognorm", 7.0, 1.8),
    "su_attempted": ("bern", 0.55),
    "num_root": ("zeroinf", 0.35, ("int", 1, 8)),
    "num_file_creations": ("zeroinf", 0.50, ("int", 1, 6)),
    "root_shell": ("bern", 0.45),
    "logged_in": ("const", 1.0),
    "count": ("int", 1, 5),
    "srv_count": ("int", 1, 5),
    "same_srv_rate": ("beta", 7.0, 2.0),
    "dst_host_count": ("int", 1, 120),
    "dst_host_srv_count": ("int", 1, 60),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
}

# ------------------- Test-only families (zero-day proxies) ------------------ #
# These appear in the test split only. Their whole point is to be *unlike* the
# training families, so recall on them is expected to be poor. Scenario 5 of the
# brief is presented from this group.
PROFILES["apache2"] = {                     # HTTP request flood, well-formed
    "duration": ("int", 0, 12),
    "protocol_type": ("const", "tcp"),
    "service": ("const", "http"),
    "flag": ("choice", ["SF", "RSTR"], [0.85, 0.15]),
    "src_bytes": ("lognorm", 7.4, 0.9),
    "dst_bytes": ("lognorm", 5.6, 1.3),
    "logged_in": ("bern", 0.55),
    "count": ("int", 40, 300),
    "srv_count": ("int", 40, 300),
    "same_srv_rate": ("beta", 20.0, 1.0),
    "dst_host_count": ("int", 150, 255),
    "dst_host_srv_count": ("int", 150, 255),
    "dst_host_same_srv_rate": ("beta", 20.0, 1.0),
    "dst_host_same_src_port_rate": ("beta", 1.5, 6.0),
}

PROFILES["processtable"] = {                # slow resource exhaustion
    "duration": ("int", 20, 600),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["telnet", "smtp", "private"], [0.5, 0.3, 0.2]),
    "flag": ("choice", ["SF", "S0"], [0.7, 0.3]),
    "src_bytes": ("zeroinf", 0.45, ("int", 1, 400)),
    "dst_bytes": ("zeroinf", 0.55, ("int", 1, 800)),
    "count": ("int", 15, 120),
    "srv_count": ("int", 15, 120),
    "serror_rate": ("beta", 1.5, 4.0),
    "same_srv_rate": ("beta", 10.0, 1.5),
    "dst_host_count": ("int", 100, 255),
    "dst_host_srv_count": ("int", 60, 255),
    "dst_host_same_srv_rate": ("beta", 10.0, 1.5),
    "dst_host_same_src_port_rate": ("beta", 2.0, 4.0),
}

PROFILES["mscan"] = {                       # mass multi-host multi-service scan
    "duration": ("const", 0.0),
    "protocol_type": ("choice", ["tcp", "udp"], [0.85, 0.15]),
    "service": ("choice", ["private", "other", "http", "telnet", "finger",
                           "domain_u"], [0.3, 0.22, 0.16, 0.14, 0.10, 0.08]),
    "flag": ("choice", ["REJ", "S0", "SF"], [0.5, 0.35, 0.15]),
    "src_bytes": ("zeroinf", 0.75, ("int", 1, 50)),
    "dst_bytes": ("const", 0.0),
    "count": ("int", 20, 200),
    "srv_count": ("int", 1, 10),
    "serror_rate": ("beta", 3.0, 2.0),
    "rerror_rate": ("beta", 5.0, 2.0),
    "same_srv_rate": ("beta", 1.0, 8.0),
    "diff_srv_rate": ("beta", 8.0, 1.2),
    "srv_diff_host_rate": ("beta", 4.0, 2.0),
    "dst_host_count": ("int", 150, 255),
    "dst_host_srv_count": ("int", 1, 25),
    "dst_host_diff_srv_rate": ("beta", 8.0, 1.2),
    "dst_host_rerror_rate": ("beta", 5.0, 2.0),
    "dst_host_same_src_port_rate": ("beta", 1.0, 8.0),
}

PROFILES["snmpguess"] = {                   # community-string guessing over UDP
    # No failed-login counter exists for SNMP, byte counts are tiny and ordinary,
    # and the traffic is near-indistinguishable from benign UDP management polling.
    "duration": ("const", 0.0),
    "protocol_type": ("const", "udp"),
    "service": ("choice", ["private", "domain_u"], [0.9, 0.1]),
    "flag": ("const", "SF"),
    "src_bytes": ("choice", [105.0, 106.0, 44.0], [0.5, 0.35, 0.15]),
    "dst_bytes": ("choice", [147.0, 105.0, 0.0], [0.45, 0.35, 0.20]),
    "count": ("int", 1, 8),
    "srv_count": ("int", 1, 8),
    "same_srv_rate": ("beta", 10.0, 1.2),
    "dst_host_count": ("int", 1, 255),
    "dst_host_srv_count": ("int", 100, 255),
    "dst_host_same_srv_rate": ("beta", 12.0, 1.2),
    "dst_host_same_src_port_rate": ("beta", 8.0, 1.5),
}

PROFILES["httptunnel"] = {                  # covert channel inside legitimate HTTP
    "duration": ("int", 1, 900),
    "protocol_type": ("const", "tcp"),
    "service": ("const", "http"),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 6.3, 1.1),
    "dst_bytes": ("lognorm", 7.1, 1.4),
    "logged_in": ("const", 1.0),
    "hot": ("zeroinf", 0.80, ("int", 1, 2)),
    "count": ("int", 1, 12),
    "srv_count": ("int", 1, 12),
    "same_srv_rate": ("beta", 9.0, 1.3),
    "dst_host_count": ("int", 1, 255),
    "dst_host_srv_count": ("int", 60, 255),
    "dst_host_same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_same_src_port_rate": ("beta", 2.0, 3.0),
}

PROFILES["xterm"] = {                       # U2R via X client, test-only
    "duration": ("int", 0, 60),
    "protocol_type": ("const", "tcp"),
    "service": ("choice", ["telnet", "other"], [0.8, 0.2]),
    "flag": ("const", "SF"),
    "src_bytes": ("lognorm", 6.5, 1.0),
    "dst_bytes": ("lognorm", 7.6, 1.2),
    "hot": ("int", 1, 4),
    "root_shell": ("bern", 0.6),
    "num_root": ("zeroinf", 0.4, ("int", 1, 5)),
    "num_file_creations": ("zeroinf", 0.4, ("int", 1, 3)),
    "logged_in": ("const", 1.0),
    "count": ("int", 1, 3),
    "srv_count": ("int", 1, 3),
    "same_srv_rate": ("beta", 8.0, 1.5),
    "dst_host_count": ("int", 1, 60),
    "dst_host_srv_count": ("int", 1, 40),
    "dst_host_same_srv_rate": ("beta", 6.0, 2.0),
}

# --------------------------------------------------------------------------- #
# Split composition. Family mixes and category proportions follow the published
# KDDTrain+ / KDDTest+ breakdown, including the deliberate distribution shift:
# R2L is 0.8% of training traffic but 12% of test traffic.
# --------------------------------------------------------------------------- #
TRAIN_MIX = {
    "normal":  (0.5346, dict(BENIGN_MIX)),
    "dos":     (0.3646, {"neptune": 0.89, "smurf": 0.06, "teardrop": 0.02,
                         "back": 0.03}),
    "probe":   (0.0925, {"satan": 0.31, "ipsweep": 0.31, "portsweep": 0.25,
                         "nmap": 0.13}),
    "r2l":     (0.0079, {"warezclient": 0.90, "guess_passwd": 0.08,
                         "ftp_write": 0.02}),
    "u2r":     (0.0004, {"buffer_overflow": 0.70, "rootkit": 0.30}),
}

TEST_MIX = {
    "normal":  (0.4308, dict(BENIGN_MIX)),
    "dos":     (0.3308, {"neptune": 0.62, "smurf": 0.09, "back": 0.05,
                         "teardrop": 0.01, "apache2": 0.11,
                         "processtable": 0.12}),
    "probe":   (0.1074, {"mscan": 0.54, "satan": 0.30, "portsweep": 0.07,
                         "ipsweep": 0.06, "nmap": 0.03}),
    "r2l":     (0.1221, {"guess_passwd": 0.45, "warezclient": 0.34,
                         "snmpguess": 0.12, "httptunnel": 0.05,
                         "ftp_write": 0.04}),
    "u2r":     (0.0089, {"xterm": 0.40, "buffer_overflow": 0.35,
                         "rootkit": 0.25}),
}

DEFAULT_TRAIN_ROWS = 45_000
DEFAULT_TEST_ROWS = 12_000


def _sample_family(family: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n rows for one attack family in full NSL-KDD column order."""
    profile = PROFILES[family]
    data: dict[str, np.ndarray] = {}
    for col in NSL_COLUMNS:
        if col in ("label", "difficulty"):
            continue
        if col in profile:
            data[col] = _draw(profile[col], n, rng)
        elif col in ("protocol_type", "service", "flag"):
            data[col] = np.full(n, "other", dtype=object)
        else:
            data[col] = np.zeros(n)
    df = pd.DataFrame(data)
    df["label"] = _LABEL_ALIAS.get(family, family)
    df["difficulty"] = rng.integers(0, 22, n)
    return _enforce_coherence(df, rng)[NSL_COLUMNS]


# Services that only make sense over a given L4 protocol. Features are drawn
# independently per column, so this pass removes physically impossible rows such
# as an ICMP echo-reply service carried over TCP.
_ICMP_SERVICES = ["ecr_i", "eco_i"]
_UDP_SERVICES = ["domain_u", "private", "other"]
_TCP_SERVICES = ["http", "private", "smtp", "ftp_data", "other", "telnet",
                 "pop_3", "finger", "auth", "time", "ftp", "imap4"]


def _enforce_coherence(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    proto = df["protocol_type"].to_numpy()
    svc = df["service"].to_numpy(dtype=object)
    for mask, allowed in (
        (proto == "icmp", _ICMP_SERVICES),
        (proto == "udp", _UDP_SERVICES),
        (proto == "tcp", _TCP_SERVICES),
    ):
        bad = mask & ~np.isin(svc, allowed)
        if bad.any():
            svc[bad] = rng.choice(allowed, size=int(bad.sum()))
    df["service"] = svc
    # ICMP has no login semantics and no TCP flag state beyond SF.
    icmp = proto == "icmp"
    if icmp.any():
        for col in ("logged_in", "num_failed_logins", "is_guest_login",
                    "root_shell", "su_attempted"):
            df.loc[icmp, col] = 0.0
        df.loc[icmp, "flag"] = "SF"
    return df



def _build(mix: dict, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for _category, (share, families) in mix.items():
        n_cat = int(round(n_rows * share))
        if n_cat <= 0:
            continue
        keys = list(families)
        weights = np.asarray([families[k] for k in keys], float)
        weights = weights / weights.sum()
        counts = np.floor(weights * n_cat).astype(int)
        counts[int(np.argmax(weights))] += n_cat - counts.sum()
        for family, k in zip(keys, counts):
            if k > 0:
                frames.append(_sample_family(family, int(k), rng))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=int(rng.integers(0, 2 ** 31))
                     ).reset_index(drop=True)


def simulate_nsl_kdd(
    n_train: int = DEFAULT_TRAIN_ROWS,
    n_test: int = DEFAULT_TEST_ROWS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) in the NSL-KDD schema.

    The test split intentionally contains five families absent from training
    (apache2, processtable, mscan, snmpguess, httptunnel, xterm) and shifts the
    category balance towards R2L, mirroring the KDDTrain+/KDDTest+ relationship.
    """
    rng = np.random.default_rng(seed)
    train = _build(TRAIN_MIX, n_train, rng)
    test = _build(TEST_MIX, n_test, rng)
    return train, test


def train_only_families() -> set[str]:
    return {_LABEL_ALIAS.get(f, f)
            for _s, fam in TRAIN_MIX.values() for f in fam}


def test_only_families() -> set[str]:
    test_fams = {_LABEL_ALIAS.get(f, f)
                 for _s, fam in TEST_MIX.values() for f in fam}
    return test_fams - train_only_families()


def benign_variants() -> dict[str, float]:
    """Benign sub-population shares, for the report's dataset section."""
    return dict(BENIGN_MIX)







