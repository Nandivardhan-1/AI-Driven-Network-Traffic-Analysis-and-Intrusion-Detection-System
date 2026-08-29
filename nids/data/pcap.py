"""Pure-Python PCAP reader and connection-feature extractor.

This is the bridge between raw capture and the detection engine: it turns a
``tcpdump``/Wireshark capture into rows with the *same* NSL-KDD feature semantics
that the models were trained on, so a model fitted on the benchmark can be pointed
at a real capture without retraining.

Implemented with ``struct`` alone -- no scapy, no dpkt -- so the repository has no
capture-library dependency. Classic pcap (linktypes Ethernet/Raw-IP/Linux-SLL) is
supported; pcapng is not.

Feature derivation follows the original KDD Cup construction:

* basic per-connection features from the packet stream itself;
* "same host" features over a trailing 2-second window;
* "same host, last 100 connections" features (the ``dst_host_*`` block).
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import NSL_COLUMNS

# Magic as seen after unpacking the first four bytes with "<I". A little-endian
# capture (what tcpdump writes on x86) stores D4 C3 B2 A1, which reads back as
# 0xA1B2C3D4; a big-endian capture reads back as 0xD4C3B2A1.
PCAP_MAGIC_LE = 0xA1B2C3D4
PCAP_MAGIC_BE = 0xD4C3B2A1
PCAP_MAGIC_NS_LE = 0xA1B23C4D
PCAP_MAGIC_NS_BE = 0x4D3CB2A1

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW_IP = 101
LINKTYPE_LINUX_SLL = 113

PORT_SERVICE = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    43: "whois", 53: "domain", 79: "finger", 80: "http", 110: "pop_3",
    111: "sunrpc", 113: "auth", 119: "nntp", 143: "imap4", 161: "private",
    389: "ldap", 443: "http_443", 445: "private", 513: "login", 514: "shell",
    515: "printer", 3306: "sql_net", 3389: "other", 8080: "http_8001",
}


@dataclass
class Packet:
    ts: float
    src: str
    dst: str
    proto: str            # tcp / udp / icmp / other
    sport: int
    dport: int
    length: int           # IP payload length
    flags: int = 0        # TCP flag bits
    icmp_type: int = -1
    frag_bad: int = 0     # non-zero fragment offset with MF cleared oddities


def read_pcap(path: str, max_packets: int | None = None) -> list[Packet]:
    """Parse a classic pcap file into a list of Packet records."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 24:
        raise ValueError(f"{path}: too short to be a pcap file")
    magic = struct.unpack("<I", raw[:4])[0]
    if magic in (PCAP_MAGIC_LE, PCAP_MAGIC_NS_LE):
        endian, nano = "<", magic == PCAP_MAGIC_NS_LE
    elif magic in (PCAP_MAGIC_BE, PCAP_MAGIC_NS_BE):
        endian, nano = ">", magic == PCAP_MAGIC_NS_BE
    else:
        raise ValueError(
            f"{path}: unsupported magic 0x{magic:08x}. pcapng is not supported; "
            "convert with `tshark -F pcap -r in.pcapng -w out.pcap`."
        )
    linktype = struct.unpack(endian + "I", raw[20:24])[0]
    off, out = 24, []
    hdr = struct.Struct(endian + "IIII")
    while off + 16 <= len(raw):
        ts_s, ts_frac, incl, _orig = hdr.unpack_from(raw, off)
        off += 16
        payload = raw[off:off + incl]
        off += incl
        ts = ts_s + ts_frac / (1e9 if nano else 1e6)
        pkt = _decode(payload, linktype, ts)
        if pkt is not None:
            out.append(pkt)
        if max_packets and len(out) >= max_packets:
            break
    return out


def _decode(buf: bytes, linktype: int, ts: float) -> Packet | None:
    """Strip the link layer, then decode IPv4/TCP/UDP/ICMP."""
    if linktype == LINKTYPE_ETHERNET:
        if len(buf) < 14:
            return None
        etype = struct.unpack(">H", buf[12:14])[0]
        offset = 14
        while etype in (0x8100, 0x88A8) and len(buf) >= offset + 4:   # VLAN tags
            etype = struct.unpack(">H", buf[offset + 2:offset + 4])[0]
            offset += 4
        if etype != 0x0800:
            return None
        ip = buf[offset:]
    elif linktype == LINKTYPE_LINUX_SLL:
        if len(buf) < 16 or struct.unpack(">H", buf[14:16])[0] != 0x0800:
            return None
        ip = buf[16:]
    elif linktype == LINKTYPE_RAW_IP:
        ip = buf
    else:
        return None

    if len(ip) < 20 or (ip[0] >> 4) != 4:
        return None
    ihl = (ip[0] & 0x0F) * 4
    total_len = struct.unpack(">H", ip[2:4])[0]
    frag_field = struct.unpack(">H", ip[6:8])[0]
    proto_num = ip[9]
    src = socket.inet_ntoa(ip[12:16])
    dst = socket.inet_ntoa(ip[16:20])
    payload = ip[ihl:]
    body_len = max(total_len - ihl, 0)

    # A non-zero fragment offset with the More-Fragments bit clear and a tiny
    # payload is the teardrop signature; surface it as wrong_fragment.
    frag_offset = frag_field & 0x1FFF
    more_frags = bool(frag_field & 0x2000)
    frag_bad = 1 if (frag_offset > 0 and not more_frags and body_len < 40) else 0

    if proto_num == 6 and len(payload) >= 14:
        sport, dport = struct.unpack(">HH", payload[:4])
        data_off = (payload[12] >> 4) * 4
        return Packet(ts, src, dst, "tcp", sport, dport,
                      max(body_len - data_off, 0), flags=payload[13],
                      frag_bad=frag_bad)
    if proto_num == 17 and len(payload) >= 8:
        sport, dport = struct.unpack(">HH", payload[:4])
        return Packet(ts, src, dst, "udp", sport, dport,
                      max(body_len - 8, 0), frag_bad=frag_bad)
    if proto_num == 1 and len(payload) >= 4:
        return Packet(ts, src, dst, "icmp", 0, 0, max(body_len - 8, 0),
                      icmp_type=payload[0], frag_bad=frag_bad)
    return Packet(ts, src, dst, "other", 0, 0, body_len, frag_bad=frag_bad)


FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


@dataclass
class Flow:
    src: str
    dst: str
    dport: int
    proto: str
    start: float
    end: float = 0.0
    src_bytes: int = 0
    dst_bytes: int = 0
    src_pkts: int = 0
    dst_pkts: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    orig_flags: int = 0
    resp_flags: int = 0
    icmp_types: set = field(default_factory=set)

    @property
    def service(self) -> str:
        if self.proto == "icmp":
            if 8 in self.icmp_types:
                return "eco_i"
            if 0 in self.icmp_types:
                return "ecr_i"
            if 3 in self.icmp_types:
                return "urp_i"
            return "other"
        svc = PORT_SERVICE.get(self.dport)
        if svc is None:
            return "domain_u" if (self.proto == "udp" and self.dport == 53) else "private"
        if self.proto == "udp" and svc == "domain":
            return "domain_u"
        return svc

    @property
    def flag(self) -> str:
        """Connection state, using the same vocabulary as NSL-KDD's ``flag``."""
        if self.proto != "tcp":
            return "SF"
        o, r = self.orig_flags, self.resp_flags
        syn_ack = bool(r & SYN) and bool(r & ACK)
        if not syn_ack:
            if r & RST:
                return "REJ"
            if o & SYN:
                return "S0"
            return "OTH"
        if (o & FIN) and (r & FIN):
            return "SF"
        if o & RST:
            return "RSTO"
        if r & RST:
            return "RSTR"
        if o & FIN:
            return "S2"
        if r & FIN:
            return "S3"
        return "S1"


def extract_flows(packets: list[Packet], timeout: float = 64.0) -> list[Flow]:
    """Group packets into bidirectional connections, ordered by start time."""
    live: dict[tuple, Flow] = {}
    done: list[Flow] = []
    for p in sorted(packets, key=lambda q: q.ts):
        a, b = (p.src, p.sport), (p.dst, p.dport)
        key = (p.proto,) + (a + b if a <= b else b + a)
        flow = live.get(key)
        if flow is not None and p.ts - flow.end > timeout:
            done.append(flow)
            flow = None
        if flow is None:
            flow = Flow(src=p.src, dst=p.dst, dport=p.dport, proto=p.proto,
                        start=p.ts, end=p.ts)
            live[key] = flow
        is_orig = (p.src == flow.src)
        if is_orig:
            flow.src_bytes += p.length
            flow.src_pkts += 1
            flow.orig_flags |= p.flags
        else:
            flow.dst_bytes += p.length
            flow.dst_pkts += 1
            flow.resp_flags |= p.flags
        flow.wrong_fragment += p.frag_bad
        flow.urgent += 1 if (p.flags & URG) else 0
        if p.icmp_type >= 0:
            flow.icmp_types.add(p.icmp_type)
        flow.end = p.ts
    done.extend(live.values())
    done.sort(key=lambda f: f.start)
    return done


_SERROR_FLAGS = {"S0", "S1", "S2", "S3"}
_RERROR_FLAGS = {"REJ", "RSTO", "RSTR"}


def flows_to_frame(flows: list[Flow], window: float = 2.0,
                   host_history: int = 100) -> pd.DataFrame:
    """Derive the NSL-KDD feature block from a list of flows.

    ``window`` is the trailing time window for the ``count``/``srv_count`` family
    and ``host_history`` the number of prior connections to the same destination
    host used for the ``dst_host_*`` family -- both matching the KDD definitions.
    """
    rows = []
    starts = np.array([f.start for f in flows]) if flows else np.zeros(0)
    for i, f in enumerate(flows):
        lo = int(np.searchsorted(starts, f.start - window, side="left"))
        recent = flows[lo:i]
        same_host = [g for g in recent if g.dst == f.dst]
        same_srv = [g for g in recent if g.service == f.service]
        host_prior = [g for g in flows[max(0, i - host_history):i] if g.dst == f.dst]
        host_srv_prior = [g for g in host_prior if g.service == f.service]
        rows.append(_feature_row(f, same_host, same_srv, host_prior, host_srv_prior))
    df = pd.DataFrame(rows)
    for col in NSL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col not in ("protocol_type", "service", "flag", "label") else "unknown"
    return df[[c for c in NSL_COLUMNS if c != "difficulty"] + ["src_ip", "dst_ip",
                                                               "dst_port", "start_time"]]


def _rate(items, predicate) -> float:
    return float(np.mean([predicate(g) for g in items])) if items else 0.0


def _feature_row(f: Flow, same_host, same_srv, host_prior, host_srv_prior) -> dict:
    return {
        "duration": round(f.end - f.start, 6),
        "protocol_type": f.proto,
        "service": f.service,
        "flag": f.flag,
        "src_bytes": float(f.src_bytes),
        "dst_bytes": float(f.dst_bytes),
        "land": 1.0 if (f.src == f.dst) else 0.0,
        "wrong_fragment": float(f.wrong_fragment),
        "urgent": float(f.urgent),
        # Content features need payload inspection, which this extractor does not
        # perform; they stay zero and the report says so explicitly.
        "count": float(len(same_host)),
        "srv_count": float(len(same_srv)),
        "serror_rate": _rate(same_host, lambda g: g.flag in _SERROR_FLAGS),
        "srv_serror_rate": _rate(same_srv, lambda g: g.flag in _SERROR_FLAGS),
        "rerror_rate": _rate(same_host, lambda g: g.flag in _RERROR_FLAGS),
        "srv_rerror_rate": _rate(same_srv, lambda g: g.flag in _RERROR_FLAGS),
        "same_srv_rate": _rate(same_host, lambda g: g.service == f.service),
        "diff_srv_rate": _rate(same_host, lambda g: g.service != f.service),
        "srv_diff_host_rate": _rate(same_srv, lambda g: g.dst != f.dst),
        "dst_host_count": float(len(host_prior)),
        "dst_host_srv_count": float(len(host_srv_prior)),
        "dst_host_same_srv_rate": _rate(host_prior, lambda g: g.service == f.service),
        "dst_host_diff_srv_rate": _rate(host_prior, lambda g: g.service != f.service),
        "dst_host_same_src_port_rate": _rate(host_prior, lambda g: g.dport == f.dport),
        "dst_host_srv_diff_host_rate": _rate(host_srv_prior, lambda g: g.dst != f.dst),
        "dst_host_serror_rate": _rate(host_prior, lambda g: g.flag in _SERROR_FLAGS),
        "dst_host_srv_serror_rate": _rate(host_srv_prior, lambda g: g.flag in _SERROR_FLAGS),
        "dst_host_rerror_rate": _rate(host_prior, lambda g: g.flag in _RERROR_FLAGS),
        "dst_host_srv_rerror_rate": _rate(host_srv_prior, lambda g: g.flag in _RERROR_FLAGS),
        "label": "unknown",
        "src_ip": f.src,
        "dst_ip": f.dst,
        "dst_port": int(f.dport),
        "start_time": f.start,
    }


def load_pcap_as_frame(path: str, max_packets: int | None = None) -> pd.DataFrame:
    """One-shot: pcap file -> NSL-KDD-shaped feature frame with flow identifiers."""
    flows = extract_flows(read_pcap(path, max_packets=max_packets))
    if not flows:
        raise ValueError(f"{path}: no IPv4 TCP/UDP/ICMP flows found")
    df = flows_to_frame(flows)
    df["category"] = "unknown"
    df["is_attack"] = 0
    return df





