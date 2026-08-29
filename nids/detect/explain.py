"""Alert rendering: the analyst-facing surface of requirement D.

Two consumers, one source of truth. ``format_alert`` writes the terminal view with
ANSI colour; the HTML dashboard consumes the same record dictionaries produced by
``HybridDetector.explain_alerts``. Nothing is computed here -- this module only
formats, so the CLI and the dashboard can never disagree about why something fired.
"""
from __future__ import annotations

import os
import sys

SEVERITY_COLOUR = {
    "critical": "\033[1;97;41m",   # white on red
    "high": "\033[1;31m",
    "medium": "\033[1;33m",
    "low": "\033[36m",
    "info": "\033[2m",
}
CATEGORY_COLOUR = {"dos": "\033[35m", "probe": "\033[36m", "r2l": "\033[33m",
                   "u2r": "\033[31m", "unknown": "\033[95m", "normal": "\033[32m"}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def colour_enabled(stream=None) -> bool:
    """Colour unless piped, or NO_COLOR is set (https://no-color.org)."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _c(text: str, code: str, enable: bool) -> str:
    return f"{code}{text}{RESET}" if enable else text


def severity_tag(severity: str, enable: bool = True) -> str:
    return _c(f" {severity.upper():^8} ", SEVERITY_COLOUR.get(severity, ""), enable)


def format_alert(rec: dict, colour: bool = True, index: int | None = None,
                 width: int = 78, verbose: bool = True) -> str:
    """Full multi-line view of one alert."""
    out = []
    ident = rec["identity"]
    who = ident.get("src_ip")
    flow = (f"{who} -> {ident.get('dst_ip', '?')}:{int(ident['dst_port'])}"
            if who and "dst_port" in ident else None)
    title = f"[{index}] " if index is not None else ""
    title += f"{rec['category'].upper()} / confidence {rec['confidence']:.2f}"
    out.append(severity_tag(rec["severity"], colour) + " " +
               _c(title, BOLD, colour))
    if flow:
        out.append(f"  flow      : {flow}")
    meta = ", ".join(f"{k}={ident[k]}" for k in
                     ("protocol_type", "service", "flag", "duration", "src_bytes",
                      "dst_bytes", "count") if k in ident)
    if meta:
        out.append(f"  connection: {meta}")
    if "ground_truth" in rec:
        gt = rec["ground_truth"]
        out.append(_c(f"  truth     : {gt.get('label', '?')} "
                      f"({gt.get('category', '?')})", DIM, colour))
    out.append(f"  layers    : {', '.join(rec['layers']) or 'none'}")

    for r in rec["rules"]:
        out.append(_c(f"  rule {r['rule_id']}   : {r['rule']} "
                      f"[{r['severity']}]", BOLD, colour))
        if verbose:
            out.append(_wrap(f"why: {r['why']}", 14, width))
            obs = ", ".join(f"{k}={v}" for k, v in r["observed"].items())
            out.append(_wrap(f"observed: {obs}", 14, width))

    if "supervised" in rec["layers"] or verbose:
        s = rec["supervised"]
        out.append(f"  model     : {s['predicted']} p={s['probability']:.3f} "
                   f"(attack p={s['attack_probability']:.3f})")
        if verbose:
            for f in s["features"][:5]:
                out.append(f"      {f['direction']:>6} {s['predicted']:<7} "
                           f"{f['contribution']:+.4f}  {f['feature']}="
                           f"{f['observed']:.4g}")

    a = rec["anomaly"]
    if a["alerted"] or verbose:
        flagword = "ALERT" if a["alerted"] else "below threshold"
        out.append(f"  anomaly   : score {a['score']:.4f} vs threshold "
                   f"{a['threshold']:.4f} ({flagword}); more unusual than "
                   f"{a['more_unusual_than']:.1%} of benign traffic")
        if verbose:
            for f in a["features"][:5]:
                out.append(f"      weight {f['weight']:.3f}  {f['feature']}="
                           f"{f['observed']:.4g}  ({f['z']:+.1f} SD vs benign mean "
                           f"{f['benign_mean']:.4g})")
    out.append(_wrap(f"verdict: {rec['corroboration']}", 14, width))
    return "\n".join(out)


def _wrap(text: str, indent: int, width: int) -> str:
    import textwrap
    pad = " " * indent
    body = textwrap.fill(text, width=max(30, width - indent),
                         initial_indent=pad, subsequent_indent=pad)
    return body


def format_alert_line(rec: dict, colour: bool = True) -> str:
    """One-line summary, for the queue listing."""
    ident = rec["identity"]
    where = (f"{ident.get('src_ip', '-'):>15} -> "
             f"{ident.get('dst_ip', '-'):<15}" if "src_ip" in ident
             else f"row {rec['row']:<8}")
    cat = _c(f"{rec['category']:<7}", CATEGORY_COLOUR.get(rec["category"], ""), colour)
    rules = ",".join(r["rule_id"] for r in rec["rules"]) or "-"
    return (f"{severity_tag(rec['severity'], colour)} {rec['confidence']:.2f} "
            f"{cat} {where} {str(ident.get('service', '-')):<10} "
            f"{str(ident.get('flag', '-')):<5} rules={rules:<12} "
            f"layers={len(rec['layers'])}/3")


def alert_table_header() -> str:
    return (f"{'SEVERITY':^10} conf {'category':<7} "
            f"{'flow / row':<35} {'service':<10} {'flag':<5} rules        layers")
