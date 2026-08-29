"""Self-contained static HTML dashboard: the analyst-facing prototype (requirement E).

Design constraints that shaped this file:

* **One file, no server, no network.** Figures are base64-inlined and the CSS and JS
  are embedded, so the output can be opened from disk, emailed, or committed as an
  artefact and still render identically. Nothing is fetched at view time.
* **No number is computed here.** The dashboard renders ``evaluation.json`` and
  ``scenarios.json``, which are produced by ``nids.evaluate`` and ``nids.scenarios``.
  The CLI reads the same two files, so the terminal and the browser cannot disagree.
* **Alerts are interactive but the data is static.** Filtering, sorting and search
  run client-side over a JSON array embedded in the page. That keeps the "select
  data, run detection, inspect flagged events" loop the brief asks for without
  introducing a web framework or an attack surface.

The provenance banner is deliberately prominent: if the corpus is simulated, the
first thing the reader sees says so.
"""
from __future__ import annotations

import base64
import json
import os

from jinja2 import Template

from ..config import ARTIFACT_DIR, CATEGORY_DESCRIPTION, FEATURE_GROUPS, FIGURE_DIR

FIGURE_CAPTIONS = {
    "fig1_ablation.png": (
        "Layer ablation. Each layer scored alone on the identical test split, then "
        "the pairwise unions, then the full hybrid. This is the evidence for the "
        "architecture: the hybrid's recall advantage comes from the "
        "signature-OR-supervised union, and the anomaly layer adds no incremental "
        "recall on this corpus."),
    "fig2_family_recall.png": (
        "Recall per attack family, sorted. Families absent from the training labels "
        "are red. The separation between the red and green groups is the "
        "generalisation gap, and it is the central limitation of this system."),
    "fig3_layer_by_family.png": (
        "Which layer catches which family. Reading down a column shows whether a "
        "family is found by hand-written logic, by the classifier, by statistical "
        "novelty, or by nothing at all."),
    "fig4_curves.png": (
        "ROC for the two scored layers and the precision-recall curve for the fused "
        "hybrid score. The marked point is the operating point actually deployed, "
        "not a threshold chosen after seeing the curve."),
    "fig5_confusion.png": (
        "Multiclass confusion for the supervised layer, row-normalised with raw "
        "counts below. The interesting cells are off-diagonal on the rare classes."),
    "fig6_severity_precision.png": (
        "Does the priority ranking work? Precision within each severity band, and "
        "precision as a function of how far down the queue an analyst reads."),
    "fig7_cost_and_queue.png": (
        "Section 5 evidence. Left: where processing time goes per stage. Right: the "
        "composition of the alert queue an analyst would actually face."),
    "fig8_importances.png": (
        "Feature importance in the supervised layer, coloured by the feature group "
        "each feature was selected from."),
    "fig9_scenarios.png": (
        "The five mandatory scenarios reduced to one number each. Red bars are "
        "costs and failures, not successes."),
}


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _collect_figures(figdir: str) -> list[dict]:
    if not os.path.isdir(figdir):
        return []
    out = []
    for name in sorted(os.listdir(figdir)):
        if not name.lower().endswith(".png"):
            continue
        out.append({"name": name,
                    "caption": FIGURE_CAPTIONS.get(name, ""),
                    "data": _b64(os.path.join(figdir, name))})
    return out


def _alert_rows(alerts: list[dict]) -> list[dict]:
    """Flatten explained alert records into the shape the client-side table wants."""
    rows = []
    for rec in alerts:
        ident = rec.get("identity", {})
        rules = rec.get("rules", [])
        sup = rec.get("supervised", {})
        ano = rec.get("anomaly", {})
        rows.append({
            "row": rec.get("row"),
            "severity": rec.get("severity", "info"),
            "confidence": rec.get("confidence", 0.0),
            "category": rec.get("category", "unknown"),
            "layers": rec.get("layers", []),
            "n_layers": len(rec.get("layers", [])),
            "src": str(ident.get("src_ip", "-")),
            "dst": str(ident.get("dst_ip", "-")),
            "port": ident.get("dst_port"),
            "service": str(ident.get("service", "-")),
            "flag": str(ident.get("flag", "-")),
            "src_bytes": ident.get("src_bytes"),
            "dst_bytes": ident.get("dst_bytes"),
            "count": ident.get("count"),
            "duration": ident.get("duration"),
            "rule_ids": [r.get("rule_id") for r in rules],
            "rules": rules,
            "truth": (rec.get("ground_truth") or {}).get("label", ""),
            "truth_cat": (rec.get("ground_truth") or {}).get("category", ""),
            "correct": bool((rec.get("ground_truth") or {}).get("category",
                                                               "normal") != "normal"),
            "corroboration": rec.get("corroboration", ""),
            "narrative": rec.get("narrative", ""),
            "sup_predicted": sup.get("predicted", ""),
            "sup_probability": sup.get("probability", 0.0),
            "sup_attack_probability": sup.get("attack_probability", 0.0),
            "sup_features": sup.get("features", [])[:5],
            "class_probabilities": sup.get("class_probabilities", {}),
            "ano_score": ano.get("score", 0.0),
            "ano_threshold": ano.get("threshold", 0.0),
            "ano_percentile": ano.get("more_unusual_than", 0.0),
            "ano_alerted": ano.get("alerted", False),
            "ano_features": ano.get("features", [])[:5],
        })
    return rows


def build_dashboard(report: dict, scenarios: list[dict] | None = None,
                    alerts: list[dict] | None = None,
                    figdir: str | None = None,
                    out_path: str | None = None,
                    title: str = "AI-Driven Network Intrusion Detection") -> str:
    """Render the dashboard to a single self-contained HTML file."""
    figdir = figdir or FIGURE_DIR
    out_path = out_path or os.path.join(ARTIFACT_DIR, "dashboard.html")
    rows = _alert_rows(alerts or [])

    ctx = {
        "title": title,
        "report": report,
        "scenarios": scenarios or [],
        "figures": _collect_figures(figdir),
        "alerts_json": json.dumps(rows),
        "n_alerts": len(rows),
        "feature_groups": FEATURE_GROUPS,
        "category_description": CATEGORY_DESCRIPTION,
        "json_dump": lambda o: json.dumps(o, indent=2, default=str),
    }
    html = Template(_TEMPLATE, trim_blocks=True, lstrip_blocks=True).render(**ctx)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
:root{color-scheme:light;--ink:#12181f;--mut:#5b6673;--line:#dfe4ea;--bg:#f6f7f9;
--card:#fff;--accent:#1f4e79;--crit:#b3202c;--high:#d2691e;--med:#b8860b;--low:#5b6673;
--good:#2e7d5b}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:var(--accent);color:#fff;padding:22px 28px}
header h1{margin:0 0 4px;font-size:21px;letter-spacing:-.2px}
header p{margin:0;opacity:.86;font-size:13px}
.banner{padding:11px 28px;font-size:13px;font-weight:600}
.banner.sim{background:#fdf3d4;color:#6b5300;border-bottom:1px solid #eadfa8}
.banner.real{background:#e6f4ec;color:#1c5a3f;border-bottom:1px solid #c3e3d1}
nav{position:sticky;top:0;z-index:20;background:var(--card);border-bottom:1px solid
var(--line);padding:0 28px;display:flex;gap:2px;overflow-x:auto}
nav button{background:none;border:0;border-bottom:2px solid transparent;padding:11px 13px;
font:inherit;font-size:13px;color:var(--mut);cursor:pointer;white-space:nowrap}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
main{padding:22px 28px 60px;max-width:1280px}
section{display:none}section.on{display:block}
h2{font-size:17px;margin:26px 0 6px;letter-spacing:-.2px}
h2:first-child{margin-top:0}
h3{font-size:14px;margin:20px 0 6px}
p.note{color:var(--mut);margin:4px 0 14px;max-width:78ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:16px 18px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:10px;
margin:12px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:650;letter-spacing:-.5px}
.kpi .l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.kpi .s{font-size:11px;color:var(--mut);margin-top:3px}
.kpi.bad .v{color:var(--crit)}.kpi.good .v{color:var(--good)}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:var(--card)}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child,th.l,td.l{text-align:left}
thead th{background:#eef1f5;font-size:11px;text-transform:uppercase;
letter-spacing:.4px;color:var(--mut);position:sticky;top:0;cursor:pointer;
white-space:nowrap}
tbody tr:hover{background:#f4f8fc}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:#f2f4f7;border:1px solid var(--line);border-radius:6px;padding:11px;
overflow:auto;font-size:11.5px;line-height:1.45}
.sev{display:inline-block;padding:1px 7px;border-radius:3px;font-size:10.5px;
font-weight:700;color:#fff;letter-spacing:.4px}
.sev.critical{background:var(--crit)}.sev.high{background:var(--high)}
.sev.medium{background:var(--med)}.sev.low{background:var(--low)}
.sev.info{background:#aab2bd}
.tag{display:inline-block;padding:1px 6px;border:1px solid var(--line);border-radius:3px;
font-size:10.5px;color:var(--mut);margin-right:3px;background:#fafbfc}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
.controls input,.controls select{font:inherit;font-size:13px;padding:6px 9px;
border:1px solid var(--line);border-radius:6px;background:var(--card)}
.controls input[type=search]{min-width:230px}
figure{margin:16px 0;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:14px}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:12.5px;color:var(--mut);margin-top:9px;max-width:88ch}
.detail{background:#fbfcfd}
.detail td{border-bottom:2px solid var(--line);padding:14px 16px;text-align:left}
.ev{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.ev h4{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;
color:var(--mut)}
.bar{position:relative;background:#eef1f5;border-radius:3px;height:14px;
overflow:hidden;min-width:60px}
.bar>i{position:absolute;left:0;top:0;bottom:0;background:var(--accent)}
.mini{font-size:11.5px;color:var(--mut)}
.narr{background:#eef4fa;border-left:3px solid var(--accent);padding:10px 12px;
border-radius:0 5px 5px 0;font-size:13px;margin-bottom:12px}
.finding{background:#fff8e8;border-left:3px solid #d9a520;padding:12px 14px;
border-radius:0 5px 5px 0;margin:12px 0;max-width:92ch}
.q{color:var(--mut);font-style:italic;margin:2px 0 12px}
.wrap{overflow:auto;max-height:74vh;border:1px solid var(--line);border-radius:8px}
.tr{cursor:pointer}
.ok{color:var(--good);font-weight:600}.no{color:var(--crit);font-weight:600}
footer{padding:20px 28px;color:var(--mut);font-size:12px;border-top:1px solid var(--line)}
</style></head><body>
{% set d = report.dataset %}
{% set ab = report.ablation %}
<header>
  <h1>{{ title }}</h1>
  <p>Hybrid signature + supervised + anomaly detection &middot; report generated
     {{ report.generated }} &middot; {{ d.test_rows|int }} test connections,
     {{ report.queue.alerts|int }} alerts</p>
</header>
{% if d.is_simulated %}
<div class="banner sim">SIMULATED CORPUS &mdash; every number on this page comes from
  <code>nids/data/simulate.py</code>, not from NSL-KDD. Place
  <code>KDDTrain+.txt</code> and <code>KDDTest+.txt</code> in <code>data/raw/</code>
  and re-run to regenerate this dashboard on the real benchmark.</div>
{% else %}
<div class="banner real">Real corpus: {{ d.name }} ({{ d.source }}).
  {{ d.note or "" }}</div>
{% endif %}
<nav>
  <button class="on" data-t="overview">Overview</button>
  <button data-t="alerts">Alert queue ({{ n_alerts }})</button>
  <button data-t="scenarios">Scenarios</button>
  <button data-t="evaluation">Evaluation</button>
  <button data-t="families">Families</button>
  <button data-t="rules">Rules</button>
  <button data-t="features">Features</button>
  <button data-t="figures">Figures</button>
  <button data-t="performance">Performance</button>
</nav>
<main>
<!-- ============================ OVERVIEW ============================ -->
<section id="overview" class="on">
  <h2>Headline result</h2>
  <p class="note">Binary attack-vs-normal detection on the held-out test split.
    The hybrid figure is the deployed configuration; the component figures beside it
    are the same three layers scored alone on the identical rows.</p>
  <div class="grid">
    <div class="kpi"><div class="l">Precision</div>
      <div class="v">{{ '%.3f'|format(ab.hybrid.precision) }}</div>
      <div class="s">{{ ab.hybrid.true_positives }} TP /
        {{ ab.hybrid.false_positives }} FP</div></div>
    <div class="kpi"><div class="l">Recall</div>
      <div class="v">{{ '%.3f'|format(ab.hybrid.recall) }}</div>
      <div class="s">{{ ab.hybrid.false_negatives }} attacks missed</div></div>
    <div class="kpi"><div class="l">F1</div>
      <div class="v">{{ '%.3f'|format(ab.hybrid.f1) }}</div>
      <div class="s">harmonic mean</div></div>
    <div class="kpi"><div class="l">Accuracy</div>
      <div class="v">{{ '%.3f'|format(ab.hybrid.accuracy) }}</div>
      <div class="s">all {{ d.test_rows }} connections</div></div>
    <div class="kpi bad"><div class="l">False-positive rate</div>
      <div class="v">{{ '%.3f'|format(ab.hybrid.false_positive_rate) }}</div>
      <div class="s">of benign traffic alerted</div></div>
    <div class="kpi"><div class="l">ROC AUC</div>
      <div class="v">{{ '%.3f'|format(report.score_auc.hybrid_roc_auc) }}</div>
      <div class="s">ranking quality, fused score</div></div>
  </div>
  <div class="finding"><strong>Read the two numbers below before the six above.</strong>
    Recall on attack families that appear in the training labels is
    {% set s5 = (scenarios | selectattr('id', 'equalto', 5) | list) %}
    {% if s5 %}
    <strong>{{ '%.1f%%'|format(s5[0].stats.recall_on_families_seen_in_training * 100) }}</strong>;
    on families absent from them it is
    <strong>{{ '%.1f%%'|format(s5[0].stats.recall_on_families_absent_from_training * 100) }}</strong>.
    {% endif %}
    A single accuracy figure averages those together and hides the only failure mode
    that matters operationally. Scenario 5 is about that gap.</div>

  <h2>Layer ablation</h2>
  <p class="note">The argument for a hybrid has to be measured against its own
    components, not asserted. Detection is a union of the three layers, so recall can
    only rise; the question is what precision it costs and which layer supplies it.</p>
  <table>
    <thead><tr><th class="l">configuration</th><th>precision</th><th>recall</th>
      <th>F1</th><th>FPR</th><th>accuracy</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
    <tbody>
    {% for key, label in [('signature_only','signature layer only'),
                          ('supervised_only','supervised layer only'),
                          ('anomaly_only','anomaly layer only'),
                          ('signature_or_anomaly','signature OR anomaly'),
                          ('signature_or_supervised','signature OR supervised'),
                          ('hybrid','hybrid (deployed)')] %}
      {% set m = ab[key] %}
      <tr{% if key == 'hybrid' %} style="font-weight:650;background:#eef4fa"{% endif %}>
        <td class="l">{{ label }}</td>
        <td>{{ '%.4f'|format(m.precision) }}</td>
        <td>{{ '%.4f'|format(m.recall) }}</td>
        <td>{{ '%.4f'|format(m.f1) }}</td>
        <td>{{ '%.4f'|format(m.false_positive_rate) }}</td>
        <td>{{ '%.4f'|format(m.accuracy) }}</td>
        <td>{{ m.true_positives }}</td><td>{{ m.false_positives }}</td>
        <td>{{ m.false_negatives }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
  <p class="note">Two honest observations from this table. The hybrid is identical to
    <em>signature OR supervised</em>, which means the anomaly layer contributes
    <strong>zero incremental recall</strong> on this corpus &mdash; it earns its place
    only by corroborating the other two for ranking, and by covering novel behaviour
    in principle. And the signature layer alone has an unacceptable standalone
    false-positive rate; it is usable here because fusion demotes uncorroborated rule
    hits rather than suppressing them.</p>

  <h2>Alert queue as an analyst would see it</h2>
  <div class="grid">
    <div class="kpi"><div class="l">Alerts raised</div>
      <div class="v">{{ '{:,}'.format(report.queue.alerts) }}</div>
      <div class="s">{{ '%.1f%%'|format(report.queue.alert_rate * 100) }} of
        {{ '{:,}'.format(report.queue.rows) }} connections</div></div>
    {% for s in report.severity %}
    <div class="kpi {% if s.precision >= 0.9 %}good{% elif s.precision < 0.7 %}bad{% endif %}">
      <div class="l">{{ s.severity }}</div>
      <div class="v">{{ '{:,}'.format(s.alerts) }}</div>
      <div class="s">precision {{ '%.3f'|format(s.precision) }},
        {{ '%.0f%%'|format(s.share_of_queue * 100) }} of queue</div></div>
    {% endfor %}
  </div>
  <p class="note">Severity is derived from the fused confidence after the
    corroboration discount, so it measures agreement between layers rather than the
    loudness of any single one. That the critical band is the most precise is the
    check that the ranking works.</p>

  <h2>Dataset</h2>
  <div class="card">
    <table>
      <tbody>
        <tr><td class="l">source</td><td class="l">{{ d.source }}</td></tr>
        <tr><td class="l">name</td><td class="l">{{ d.name }}</td></tr>
        <tr><td class="l">training rows</td><td class="l">{{ '{:,}'.format(d.train_rows) }}</td></tr>
        <tr><td class="l">test rows</td><td class="l">{{ '{:,}'.format(d.test_rows) }}</td></tr>
        <tr><td class="l">train class mix</td><td class="l">
          {% for k, v in d.train_categories.items() %}<span class="tag">{{ k }}:
          {{ '{:,}'.format(v) }}</span>{% endfor %}</td></tr>
        <tr><td class="l">test class mix</td><td class="l">
          {% for k, v in d.test_categories.items() %}<span class="tag">{{ k }}:
          {{ '{:,}'.format(v) }}</span>{% endfor %}</td></tr>
        <tr><td class="l">families absent from training</td><td class="l">
          {% for f in d.families_absent_from_training %}<span class="tag"
          style="color:var(--crit);border-color:#eebfc3">{{ f }}</span>{% endfor %}</td></tr>
      </tbody>
    </table>
    {% if d.note %}<p class="note" style="margin-bottom:0">{{ d.note }}</p>{% endif %}
  </div>
</section>
<!-- ============================ ALERTS ============================ -->
<section id="alerts">
  <h2>Alert queue</h2>
  <p class="note">The top {{ n_alerts }} alerts by fused confidence, each fully
    explained. Click any row to expand the evidence: the rule that matched with the
    values observed, the supervised model's per-feature contributions, and the
    anomaly layer's most isolating features with their distance from the benign mean.
    This is requirement D &mdash; no row is a bare label.</p>
  <div class="controls">
    <input type="search" id="q" placeholder="search ip, service, rule, family...">
    <select id="fsev"><option value="">all severities</option>
      <option>critical</option><option>high</option><option>medium</option>
      <option>low</option></select>
    <select id="fcat"><option value="">all categories</option></select>
    <select id="flay"><option value="">any corroboration</option>
      <option value="3">3 of 3 layers</option><option value="2">2 of 3 layers</option>
      <option value="1">1 layer only</option></select>
    <select id="fver"><option value="">true and false positives</option>
      <option value="tp">true positives only</option>
      <option value="fp">false positives only</option></select>
    <span class="mini" id="cnt"></span>
  </div>
  <div class="wrap"><table id="tbl">
    <thead><tr>
      <th class="l" data-k="severity">severity</th>
      <th data-k="confidence">conf</th>
      <th class="l" data-k="category">category</th>
      <th class="l" data-k="src">source</th>
      <th class="l" data-k="dst">destination</th>
      <th class="l" data-k="service">service</th>
      <th class="l" data-k="flag">flag</th>
      <th data-k="count">count</th>
      <th class="l" data-k="rule_ids">rules</th>
      <th data-k="n_layers">layers</th>
      <th class="l" data-k="truth">ground truth</th>
    </tr></thead><tbody id="tb"></tbody>
  </table></div>
</section>

<!-- ============================ SCENARIOS ============================ -->
<section id="scenarios">
  <h2>Demonstration scenarios</h2>
  <p class="note">The five scenarios the brief requires, plus one extra that runs the
    whole pipeline on a raw packet capture. Scenario 1 reports a cost and scenario 5
    reports a failure; neither is presented as a success.</p>
  {% for sc in scenarios %}
  <div class="card">
    <h3 style="margin-top:0">Scenario {{ sc.id }} &mdash; {{ sc.title }}</h3>
    {% if sc.question %}<p class="q">{{ sc.question }}</p>{% endif %}
    <table><tbody>
    {% for k, v in sc.stats.items() %}
      {% if v is mapping %}
      <tr><td class="l" style="vertical-align:top">{{ k }}</td><td class="l">
        {% for k2, v2 in v.items() %}<span class="tag">{{ k2 }}:
          {% if v2 is mapping %}{{ v2.get('hybrid', v2) }}{% else %}{{ v2 }}{% endif %}
        </span>{% endfor %}</td></tr>
      {% elif v is sequence and v is not string %}
      <tr><td class="l" style="vertical-align:top">{{ k }}</td><td class="l">
        <pre style="margin:0">{{ json_dump(v) }}</pre></td></tr>
      {% else %}
      <tr><td class="l">{{ k }}</td><td class="l"><code>{{ v }}</code></td></tr>
      {% endif %}
    {% endfor %}
    </tbody></table>
    {% if sc.rules_involved %}
    <h3>rules that fired on this slice</h3>
    <p class="l">{% for rid, n in sc.rules_involved.items() %}<span class="tag">{{ rid }}
      &times;{{ '{:,}'.format(n) }}</span>{% endfor %}</p>
    {% endif %}
    {% if sc.examples %}
    <h3>explained example{{ 's' if sc.examples|length > 1 else '' }}</h3>
    {% for rec in sc.examples %}
    <div class="card" style="margin:8px 0;background:#fbfcfd">
      <div><span class="sev {{ rec.severity }}">{{ rec.severity|upper }}</span>
        <strong style="margin-left:8px">{{ rec.category|upper }}</strong>
        <span class="mini">confidence {{ '%.2f'|format(rec.confidence) }} &middot;
        layers {{ rec.layers|length }}/3
        {% if rec.ground_truth %}&middot; truth
        <code>{{ rec.ground_truth.label }}</code>{% endif %}</span></div>
      <div class="narr" style="margin-top:9px">{{ rec.narrative }}</div>
      <div class="ev">
        {% if rec.rules %}
        <div><h4>rule evidence</h4>
        {% for r in rec.rules %}
          <div class="mini" style="margin-bottom:7px">
            <strong>{{ r.rule_id }} {{ r.rule }}</strong> [{{ r.severity }}]<br>
            {{ r.why }}<br>
            observed: {% for k, v in r.observed.items() %}<code>{{ k }}={{ v }}</code>
            {% endfor %}</div>
        {% endfor %}</div>
        {% endif %}
        <div><h4>supervised contributions</h4>
          <div class="mini">predicts <strong>{{ rec.supervised.predicted }}</strong>
            at {{ '%.1f%%'|format(rec.supervised.probability * 100) }}
            (attack probability
            {{ '%.3f'|format(rec.supervised.attack_probability) }})</div>
          <table style="margin-top:5px"><tbody>
          {% for f in rec.supervised.features[:5] %}
            <tr><td class="l mini">{{ f.feature }}</td>
              <td class="mini">{{ '%.4g'|format(f.observed) }}</td>
              <td class="mini">{{ f.direction }}
                {{ '%+.4f'|format(f.contribution) }}</td></tr>
          {% endfor %}
          </tbody></table></div>
        <div><h4>anomaly attribution</h4>
          <div class="mini">score {{ '%.4f'|format(rec.anomaly.score) }} vs threshold
            {{ '%.4f'|format(rec.anomaly.threshold) }}
            &mdash; {{ 'ALERT' if rec.anomaly.alerted else 'below threshold' }};
            more unusual than
            {{ '%.1f%%'|format(rec.anomaly.more_unusual_than * 100) }} of benign
            traffic</div>
          <table style="margin-top:5px"><tbody>
          {% for f in rec.anomaly.features[:5] %}
            <tr><td class="l mini">{{ f.feature }}</td>
              <td class="mini">{{ '%.4g'|format(f.observed) }}</td>
              <td class="mini">{{ '%+.1f'|format(f.z) }} SD</td></tr>
          {% endfor %}
          </tbody></table></div>
      </div>
    </div>
    {% endfor %}
    {% endif %}
    <div class="finding"><strong>Finding.</strong> {{ sc.finding }}</div>
  </div>
  {% endfor %}
</section>
<!-- ============================ EVALUATION ============================ -->
<section id="evaluation">
  <h2>Concrete true positives, false positives and false negatives</h2>
  <p class="note">Aggregate metrics cannot show whether an alert would have been
    actionable, so the brief asks for examples. False negatives here are ranked by how
    close they came to the anomaly threshold, so these are the informative near-misses
    rather than arbitrary rows.</p>
  {% for kind, label, cls in [('true_positive','True positives','good'),
                              ('false_positive','False positives','bad'),
                              ('false_negative','False negatives (missed attacks)','bad')] %}
  <h3>{{ label }}</h3>
  {% for rec in report.examples[kind] %}
  <div class="card" style="background:#fbfcfd">
    <div><span class="sev {{ rec.severity }}">{{ rec.severity|upper }}</span>
      <strong style="margin-left:8px">{{ rec.category|upper }}</strong>
      <span class="mini">confidence {{ '%.2f'|format(rec.confidence) }}
      {% if rec.ground_truth %}&middot; truth
        <code>{{ rec.ground_truth.label }}</code>
        ({{ rec.ground_truth.category }}){% endif %}</span></div>
    <div class="narr" style="margin-top:9px">{{ rec.narrative }}</div>
  </div>
  {% endfor %}
  {% endfor %}

  <h2>Multiclass attack attribution (supervised layer)</h2>
  <div class="grid">
    <div class="kpi"><div class="l">Accuracy</div>
      <div class="v">{{ '%.3f'|format(report.multiclass.accuracy) }}</div>
      <div class="s">5-way class assignment</div></div>
    <div class="kpi"><div class="l">Macro F1</div>
      <div class="v">{{ '%.3f'|format(report.multiclass.macro_f1) }}</div>
      <div class="s">unweighted by support</div></div>
    <div class="kpi"><div class="l">OOB accuracy</div>
      <div class="v">{{ '%.4f'|format(report.fit.supervised_oob_accuracy) }}</div>
      <div class="s">out-of-bag, training data</div></div>
  </div>
  <p class="note">The gap between out-of-bag accuracy on training data and macro F1 on
    the test split is not overfitting in the usual sense: it is distribution shift.
    The test split contains families the training split does not.</p>
  <table>
    <thead><tr><th class="l">class</th><th class="l">meaning</th><th>precision</th>
      <th>recall</th><th>F1</th><th>support</th></tr></thead>
    <tbody>
    {% for r in report.multiclass.per_class %}
      <tr><td class="l"><code>{{ r.class }}</code></td>
        <td class="l mini">{{ category_description.get(r.class, '') }}</td>
        <td>{{ '%.3f'|format(r.precision) }}</td>
        <td>{{ '%.3f'|format(r.recall) }}</td>
        <td>{{ '%.3f'|format(r.f1) }}</td>
        <td>{{ '{:,}'.format(r.support) }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
  <h3>Confusion matrix</h3>
  <pre>{{ report.multiclass.confusion_text }}</pre>

  <h2>Queue quality: precision at k</h2>
  <p class="note">If an analyst reads the queue from the top and stops after k alerts,
    this is the fraction that were real. It is the metric that decides whether the
    ranking is worth anything.</p>
  <table>
    <thead><tr><th>k</th><th>precision@k</th><th>true positives in top k</th></tr></thead>
    <tbody>
    {% for p in report.precision_at_k %}
      <tr><td>{{ '{:,}'.format(p.k) }}</td>
        <td>{{ '%.4f'|format(p.precision) }}</td>
        <td>{{ '{:,}'.format(p.true_positives) }}</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Score separability</h2>
  <table>
    <thead><tr><th class="l">score</th><th>ROC AUC</th></tr></thead>
    <tbody>
      <tr><td class="l">anomaly (isolation depth)</td>
        <td>{{ '%.4f'|format(report.score_auc.anomaly_roc_auc) }}</td></tr>
      <tr><td class="l">supervised attack probability</td>
        <td>{{ '%.4f'|format(report.score_auc.supervised_roc_auc) }}</td></tr>
      <tr><td class="l">hybrid fused confidence</td>
        <td>{{ '%.4f'|format(report.score_auc.hybrid_roc_auc) }}</td></tr>
    </tbody>
  </table>
</section>

<!-- ============================ FAMILIES ============================ -->
<section id="families">
  <h2>Recall by attack family, per layer</h2>
  <p class="note">The single most diagnostic table in the evaluation. Families marked
    <span class="tag" style="color:var(--crit);border-color:#eebfc3">absent</span> never
    appear in the training labels, so they stand in for a zero-day. Reading across a
    row shows which layer, if any, caught that family.</p>
  <table>
    <thead><tr><th class="l">family</th><th class="l">category</th><th>support</th>
      <th class="l">in training</th><th>hybrid</th><th>signature</th>
      <th>supervised</th><th>anomaly</th></tr></thead>
    <tbody>
    {% for f in report.families %}
      <tr>
        <td class="l"><code>{{ f.family }}</code></td>
        <td class="l mini">{{ f.category }}</td>
        <td>{{ '{:,}'.format(f.support) }}</td>
        <td class="l">{% if f.in_training %}<span class="tag">seen</span>
          {% else %}<span class="tag"
          style="color:var(--crit);border-color:#eebfc3">absent</span>{% endif %}</td>
        <td class="{% if f.hybrid_recall >= 0.9 %}ok{% elif f.hybrid_recall < 0.5 %}no{% endif %}">
          {{ '%.3f'|format(f.hybrid_recall) }}</td>
        <td>{{ '%.3f'|format(f.signature_recall) }}</td>
        <td>{{ '%.3f'|format(f.supervised_recall) }}</td>
        <td>{{ '%.3f'|format(f.anomaly_recall) }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
</section>

<!-- ============================ RULES ============================ -->
<section id="rules">
  <h2>Signature layer coverage</h2>
  <p class="note">Every rule measured on the test split: how often it fired, how often
    that was on an actual attack, and how often it also assigned the correct attack
    category. A rule with high fire count and low precision is a false-positive source;
    a rule that never fires is dead weight and is reported as such rather than removed
    quietly.</p>
  <table>
    <thead><tr><th class="l">rule</th><th class="l">name</th><th class="l">category</th>
      <th class="l">severity</th><th>fired</th><th>on attack</th><th>on benign</th>
      <th>precision</th><th>exact category</th></tr></thead>
    <tbody>
    {% for r in report.rule_coverage %}
      <tr><td class="l"><code>{{ r.rule_id }}</code></td>
        <td class="l">{{ r.name }}</td>
        <td class="l mini">{{ r.category }}</td>
        <td class="l"><span class="sev {{ r.severity }}">{{ r.severity|upper }}</span></td>
        <td>{{ '{:,}'.format(r.fired) }}</td>
        <td>{{ '{:,}'.format(r.on_attack) }}</td>
        <td class="{% if r.on_benign > 0 %}no{% endif %}">{{ '{:,}'.format(r.on_benign) }}</td>
        <td>{{ '%.3f'|format(r.precision) }}</td>
        <td>{{ '%.3f'|format(r.exact_category) }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
</section>

<!-- ============================ FEATURES ============================ -->
<section id="features">
  <h2>Feature groups and why each was selected</h2>
  <p class="note">Requirement B asks for justification, not a list. Each group below
    exists because a specific attack behaviour is only visible through it.</p>
  {% for name, g in feature_groups.items() %}
  <div class="card">
    <h3 style="margin-top:0"><code>{{ name }}</code></h3>
    <p class="note" style="margin-bottom:8px">{{ g.why }}</p>
    <div>{% for f in g.features %}<span class="tag">{{ f }}</span>{% endfor %}</div>
  </div>
  {% endfor %}
  <h2>Measured importance in the supervised layer</h2>
  <table>
    <thead><tr><th class="l">feature</th><th class="l">group</th><th>importance</th>
      <th class="l">why this feature is in the model</th></tr></thead>
    <tbody>
    {% for r in report.importances %}
      <tr><td class="l"><code>{{ r.feature }}</code></td>
        <td class="l mini">{{ r.get('group', '') }}</td>
        <td>{{ '%.4f'|format(r.importance) }}</td>
        <td class="l mini">{{ r.get('why', '') }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
</section>

<!-- ============================ FIGURES ============================ -->
<section id="figures">
  <h2>Figures</h2>
  <p class="note">Every figure is a function of <code>evaluation.json</code>, so a
    chart cannot disagree with the table beside it.</p>
  {% for f in figures %}
  <figure>
    <img src="data:image/png;base64,{{ f.data }}" alt="{{ f.name }}">
    <figcaption><strong>{{ f.name }}</strong> &mdash; {{ f.caption }}</figcaption>
  </figure>
  {% endfor %}
</section>

<!-- ============================ PERFORMANCE ============================ -->
<section id="performance">
  <h2>Throughput and the cost of inline detection</h2>
  <p class="note">Section 5 of the brief asks what running this inline would cost.
    These figures come from the same run that produced the metrics above, on this
    machine, single-threaded, in pure NumPy with no compiled extensions.</p>
  <div class="grid">
    <div class="kpi"><div class="l">Throughput</div>
      <div class="v">{{ '{:,.0f}'.format(report.performance.rows_per_second) }}</div>
      <div class="s">connections / second, detection only</div></div>
    <div class="kpi"><div class="l">Wall time</div>
      <div class="v">{{ '%.2f'|format(report.performance.wall_seconds) }}s</div>
      <div class="s">for {{ '{:,}'.format(report.performance.test_rows) }} connections</div></div>
    <div class="kpi"><div class="l">Feature count</div>
      <div class="v">{{ report.fit.n_features }}</div>
      <div class="s">after encoding</div></div>
  </div>
  <h3>Per stage</h3>
  <table>
    <thead><tr><th class="l">stage</th><th>seconds</th></tr></thead>
    <tbody>
    {% for k, v in report.performance.per_stage_seconds.items() %}
      <tr><td class="l"><code>{{ k }}</code></td><td>{{ '%.4f'|format(v) }}</td></tr>
    {% endfor %}
    </tbody>
  </table>
  <p class="note">{{ report.performance.explanation_cost_note }}
    The anomaly layer dominates, which is the expected shape: isolation-forest scoring
    walks {{ report.fit.anomaly_percentile and '' }}every tree for every row and is not
    vectorised across trees here. A production deployment would evaluate the cheap
    signature layer inline and the two model layers out-of-band on a mirrored feed,
    which keeps the forwarding path at the signature layer's
    cost.</p>
  <h3>Fitted configuration</h3>
  <pre>{{ json_dump(report.fit) }}</pre>
</section>
</main>
<footer>Generated by <code>nids.report.html</code> from
  <code>artifacts/evaluation.json</code> and <code>artifacts/scenarios.json</code>.
  No value on this page is computed at render time.</footer>
<script>
const ALERTS = {{ alerts_json|safe }};
const SEVR = {critical:4, high:3, medium:2, low:1, info:0};
let sortKey = "confidence", sortDir = -1, openRow = null;

// ---- tabs ---------------------------------------------------------------- //
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.classList.remove("on"));
  document.querySelectorAll("main section").forEach(x => x.classList.remove("on"));
  b.classList.add("on");
  document.getElementById(b.dataset.t).classList.add("on");
  window.scrollTo({top: 0});
});

// ---- category filter options, derived from the data --------------------- //
(() => {
  const sel = document.getElementById("fcat");
  [...new Set(ALERTS.map(a => a.category))].sort().forEach(c => {
    const o = document.createElement("option"); o.textContent = c; sel.appendChild(o);
  });
})();

const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const num = (v, d = 4) => (v == null || v === "") ? "-" :
  (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(d)) : v);

function filtered() {
  const q = document.getElementById("q").value.trim().toLowerCase();
  const sev = document.getElementById("fsev").value;
  const cat = document.getElementById("fcat").value;
  const lay = document.getElementById("flay").value;
  const ver = document.getElementById("fver").value;
  return ALERTS.filter(a => {
    if (sev && a.severity !== sev) return false;
    if (cat && a.category !== cat) return false;
    if (lay && String(a.n_layers) !== lay) return false;
    if (ver === "tp" && !a.correct) return false;
    if (ver === "fp" && a.correct) return false;
    if (q) {
      const hay = [a.src, a.dst, a.service, a.flag, a.category, a.truth,
                   a.rule_ids.join(" "), a.layers.join(" "),
                   a.sup_predicted].join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  }).sort((x, y) => {
    let a = x[sortKey], b = y[sortKey];
    if (sortKey === "severity") { a = SEVR[a]; b = SEVR[b]; }
    if (sortKey === "rule_ids") { a = a.join(); b = b.join(); }
    if (a === b) return y.confidence - x.confidence;
    return (a > b ? 1 : -1) * sortDir;
  });
}

function featTable(rows, cols) {
  if (!rows || !rows.length) return '<div class="mini">none</div>';
  return '<table style="margin-top:5px"><tbody>' + rows.map(f =>
    "<tr>" + cols.map(c => `<td class="${c.l ? 'l ' : ''}mini">${c.f(f)}</td>`).join("")
    + "</tr>").join("") + "</tbody></table>";
}

function detail(a) {
  const rules = a.rules.length ? a.rules.map(r => `
      <div class="mini" style="margin-bottom:8px">
        <strong>${esc(r.rule_id)} ${esc(r.rule)}</strong> [${esc(r.severity)}]<br>
        ${esc(r.why)}<br>observed: ` +
        Object.entries(r.observed || {}).map(([k, v]) =>
          `<code>${esc(k)}=${esc(v)}</code>`).join(" ") + "</div>").join("")
    : '<div class="mini">no rule matched this connection</div>';

  const probs = Object.entries(a.class_probabilities || {})
    .sort((p, q) => q[1] - p[1]).map(([c, p]) => `
      <tr><td class="l mini">${esc(c)}</td><td class="mini">${p.toFixed(3)}</td>
      <td style="width:110px"><div class="bar"><i style="width:${(p * 100).toFixed(1)}%">
      </i></div></td></tr>`).join("");

  return `<tr class="detail"><td colspan="11">
    <div class="narr">${esc(a.narrative)}</div>
    <div class="ev">
      <div><h4>connection</h4><div class="mini">
        ${esc(a.src)} &rarr; ${esc(a.dst)}${a.port != null ? ":" + a.port : ""}<br>
        service <code>${esc(a.service)}</code>, state <code>${esc(a.flag)}</code><br>
        duration ${num(a.duration, 2)}s, src_bytes ${num(a.src_bytes, 0)},
        dst_bytes ${num(a.dst_bytes, 0)}, count ${num(a.count, 0)}<br>
        ${a.truth ? `ground truth <code>${esc(a.truth)}</code>
          (${esc(a.truth_cat)}) &mdash;
          <span class="${a.correct ? 'ok' : 'no'}">
          ${a.correct ? "true positive" : "FALSE POSITIVE"}</span>` : ""}
      </div></div>
      <div><h4>rule evidence</h4>${rules}</div>
      <div><h4>supervised model</h4>
        <div class="mini">predicts <strong>${esc(a.sup_predicted)}</strong> at
          ${(a.sup_probability * 100).toFixed(1)}% &middot; attack probability
          ${a.sup_attack_probability.toFixed(3)}</div>
        <table style="margin-top:5px"><tbody>${probs}</tbody></table>
        <div class="mini" style="margin-top:7px">largest per-feature contributions
          (additive, sum to the predicted probability)</div>
        ${featTable(a.sup_features, [
          {l: 1, f: f => "<code>" + esc(f.feature) + "</code>"},
          {f: f => Number(f.observed).toPrecision(4)},
          {f: f => esc(f.direction) + " " +
                   (f.contribution >= 0 ? "+" : "") + f.contribution.toFixed(4)}])}
      </div>
      <div><h4>anomaly layer</h4>
        <div class="mini">score ${a.ano_score.toFixed(4)} vs threshold
          ${a.ano_threshold.toFixed(4)} &mdash;
          <strong>${a.ano_alerted ? "ALERT" : "below threshold"}</strong><br>
          more unusual than ${(a.ano_percentile * 100).toFixed(1)}% of benign
          training traffic</div>
        ${featTable(a.ano_features, [
          {l: 1, f: f => "<code>" + esc(f.feature) + "</code>"},
          {f: f => Number(f.observed).toPrecision(4)},
          {f: f => (f.z >= 0 ? "+" : "") + f.z.toFixed(1) + " SD"}])}
      </div>
    </div>
    <div class="finding" style="margin-bottom:0"><strong>Corroboration.</strong>
      ${esc(a.corroboration)}</div>
  </td></tr>`;
}

function render() {
  const rows = filtered();
  document.getElementById("cnt").textContent =
    `${rows.length.toLocaleString()} of ${ALERTS.length.toLocaleString()} shown` +
    `  ·  ${rows.filter(r => r.correct).length.toLocaleString()} true positives` +
    `, ${rows.filter(r => !r.correct).length.toLocaleString()} false positives`;
  document.getElementById("tb").innerHTML = rows.map(a => `
    <tr class="tr" data-r="${a.row}">
      <td class="l"><span class="sev ${a.severity}">${a.severity.toUpperCase()}</span></td>
      <td>${a.confidence.toFixed(2)}</td>
      <td class="l">${esc(a.category)}</td>
      <td class="l mini">${esc(a.src)}</td>
      <td class="l mini">${esc(a.dst)}${a.port != null ? ":" + a.port : ""}</td>
      <td class="l"><code>${esc(a.service)}</code></td>
      <td class="l"><code>${esc(a.flag)}</code></td>
      <td>${num(a.count, 0)}</td>
      <td class="l mini">${a.rule_ids.length ? a.rule_ids.join(", ") : "&ndash;"}</td>
      <td>${a.n_layers}/3</td>
      <td class="l mini ${a.correct ? 'ok' : 'no'}">
        ${esc(a.truth || "?")}</td>
    </tr>` + (openRow === a.row ? detail(a) : "")).join("");
  document.querySelectorAll("#tb tr.tr").forEach(tr => tr.onclick = () => {
    const r = Number(tr.dataset.r);
    openRow = (openRow === r) ? null : r;
    render();
  });
}

document.querySelectorAll("#tbl thead th").forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  if (!k) return;
  if (sortKey === k) sortDir *= -1;
  else { sortKey = k; sortDir = (k === "src" || k === "dst" || k === "service" ||
                                k === "category" || k === "flag") ? 1 : -1; }
  render();
});
["q", "fsev", "fcat", "flay", "fver"].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener(id === "q" ? "input" : "change", () => { openRow = null; render(); });
});
render();
</script>
</body></html>
"""
