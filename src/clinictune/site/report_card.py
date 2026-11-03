"""Printable HTML clinic report card (self-contained: no external assets, fonts or scripts)."""

from __future__ import annotations

from html import escape
from typing import Any

from clinictune import DISCLAIMER

VERDICT = {
    "ready": ("Ready for supervised use", "#1F6B49", "#E2F1E8"),
    "use_with_caution": ("Use with caution", "#8F5A00", "#FFF0C2"),
    "not_ready": ("Not ready: do not rely on its results", "#B4232A", "#FBE6E3"),
}
TASK = {"tb": "Chest X-ray, tuberculosis", "dr": "Fundus photo, diabetic retinopathy"}
STRATEGY = {
    "recalibrate": "Recalibrated the training model",
    "adapt": "Added a local adaptation layer",
}


def _pct(v: Any) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def _ci(ci: Any) -> str:
    return "" if not ci else f"<small>{ci[0] * 100:.0f} to {ci[1] * 100:.0f}%</small>"


def _num(v: Any, d: int = 2) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def render(report: dict[str, Any]) -> str:
    r = report
    label, colour, tint = VERDICT.get(r["verdict"], (r["verdict"], "#444", "#eee"))
    before = r["before_recalibration"]
    after = r["after_recalibration_crossfit"]
    bm, bc = before["metrics"], before["conformal_coverage"]
    am = after.get("metrics", {})
    ac = after.get("conformal_coverage", {})
    target = bc.get("target", 0.9)

    def cov(v: Any) -> str:
        if v is None:
            return "–"
        cls = "ok" if v >= target - 0.05 else "bad"
        return f'<span class="{cls}">{_pct(v)}</span>'

    rows = [
        (
            "Sensitivity",
            f"{_pct(bm['sensitivity'])}{_ci(bm['sensitivity_ci'])}",
            f"{_pct(am.get('sensitivity'))}{_ci(am.get('sensitivity_ci'))}",
        ),
        (
            "Specificity",
            f"{_pct(bm['specificity'])}{_ci(bm['specificity_ci'])}",
            f"{_pct(am.get('specificity'))}{_ci(am.get('specificity_ci'))}",
        ),
        ("ROC-AUC", _num(bm.get("roc_auc")), _num(am.get("roc_auc"))),
        ("PR-AUC", _num(bm.get("pr_auc")), _num(am.get("pr_auc"))),
        ("Calibration error (ECE, lower is better)", _num(bm.get("ece")), _num(am.get("ece"))),
        (f"Confidence check holds for positives (target {target:.0%})", cov(bc.get("class_1")), cov(ac.get("class_1"))),
        (f"Confidence check holds for negatives (target {target:.0%})", cov(bc.get("class_0")), cov(ac.get("class_0"))),
    ]
    table = "".join(f"<tr><th scope='row'>{escape(a)}</th><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)

    strategies = r.get("strategies") or {}
    strat_rows = ""
    for key in ("recalibrate", "adapt"):
        s = strategies.get(key)
        if not s:
            continue
        chosen = " class='chosen'" if r.get("strategy") == key else ""
        if s.get("available"):
            m = s["metrics"]
            cells = f"<td>{_num(m.get('roc_auc'))}</td><td>{_pct(m.get('sensitivity'))}</td><td>{_pct(m.get('specificity'))}</td>"
        else:
            cells = f"<td colspan='3'>{escape(str(s.get('reason', 'not available')))}</td>"
        mark = " (in use)" if chosen else ""
        strat_rows += f"<tr{chosen}><th scope='row'>{escape(STRATEGY[key])}{mark}</th>{cells}</tr>"

    shift = r["shift"]
    gate = r.get("quality_gate", {})
    warns = "".join(f"<li>{escape(w)}</li>" for w in r.get("warnings", []))
    reasons = "".join(f"<li>{escape(w)}</li>" for w in r.get("verdict_reasons", []))
    dep = r["deployed_calibration"]
    qhat = dep["conformal"]["qhat"]
    q_text = ", ".join(
        f"{'positives' if k == '1' else 'negatives'} {'never excluded' if v is None else f'{v:.2f}'}"
        for k, v in sorted(qhat.items())
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clinic report card: {escape(r["site_id"])}</title>
<style>
:root{{--ink:#1B2A41;--ink2:#48546A;--ink3:#6B7686;--rule:#D2D8D0;--paper:#FBFCFA;--violet:#5B3E96}}
*{{box-sizing:border-box}}
body{{font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);background:var(--paper);max-width:880px;margin:0 auto;padding:28px 22px 48px}}
h1{{font-size:26px;line-height:1.2;margin:0 0 4px;font-weight:700;letter-spacing:-.01em}}
h2{{font-size:17px;margin:30px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--ink)}}
.sub{{color:var(--ink2);margin:0}}
.verdict{{display:flex;gap:16px;align-items:flex-start;margin:22px 0 6px;padding:16px 18px;border-radius:10px;background:{tint};border-left:6px solid {colour}}}
.verdict b{{font-size:20px;color:{colour};display:block;line-height:1.25}}
.verdict ul{{margin:6px 0 0;padding-left:18px;color:var(--ink2)}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--rule);vertical-align:top}}
thead th{{color:var(--ink3);font-weight:600;border-bottom:2px solid #B9C1B6}}
td small{{display:block;color:var(--ink3);font-size:12px}}
tbody th{{font-weight:500;color:var(--ink2)}}
.ok{{color:#1F6B49;font-weight:700}} .bad{{color:#B4232A;font-weight:700}}
tr.chosen th,tr.chosen td{{background:#EBE5F5}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0 22px}}
.kv div{{padding:10px 0;border-bottom:1px solid var(--rule)}} .kv b{{display:block;font-size:20px;font-weight:650}}
.kv span{{color:var(--ink3);font-size:13px}}
.disclaimer{{font-size:13px;color:var(--ink2);border:1px dashed #B9C1B6;border-radius:8px;padding:10px 12px;margin-bottom:22px}}
.note{{font-size:13px;color:var(--ink3);margin-top:8px}}
ul.warn{{padding-left:18px;color:var(--ink2)}}
.sign{{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:40px}}
.sign div{{border-top:1px solid var(--ink);padding-top:6px;color:var(--ink3);font-size:13px}}
@media print{{body{{padding:0;background:#fff}} .verdict{{break-inside:avoid}}}}
</style></head><body>
<p class="disclaimer"><b>Screening aid, decision support only.</b> {escape(DISCLAIMER)}</p>
<h1>Clinic report card: {escape(r["site_id"])}</h1>
<p class="sub">{escape(TASK.get(r["task"], r["task"]))}. Settings version {r.get("version", "?")}, model bundle {escape(r["bundle_id"])}, {escape(r["created_at"][:16].replace("T", " "))} UTC.</p>
<div class="verdict"><div><b>{escape(label)}</b><ul>{reasons}</ul></div></div>
<p class="note">{escape(r.get("strategy_reason", ""))}</p>

<h2>The images this clinic provided</h2>
<div class="kv">
<div><b>{r["counts"]["n"]}</b><span>labelled images used</span></div>
<div><b>{r["counts"]["n_pos"]} and {r["counts"]["n_neg"]}</b><span>confirmed positive and negative</span></div>
<div><b>{_pct(gate.get("pass_rate"))}</b><span>passed the quality check ({gate.get("passed", "–")} of {gate.get("submitted", "–")})</span></div>
</div>

<h2>How different this clinic is from the training hospitals</h2>
<div class="kv">
<div><b>{escape(str(shift.get("severity")))}</b><span>shift</span></div>
<div><b>{_num(shift.get("domain_auc"))}</b><span>how easily a model can tell this clinic's images apart (0.5 means it cannot)</span></div>
<div><b>{_num(shift.get("p_value"), 3)}</b><span>MMD test p-value</span></div>
<div><b>{_pct(shift.get("observed_prevalence"))}</b><span>positive here, versus {_pct(shift.get("training_prevalence"))} in training</span></div>
</div>

<h2>Before and after fitting this clinic's settings</h2>
<table><thead><tr><th></th><th>Training hospitals' settings</th><th>This clinic's settings, cross-fitted over {after.get("folds", "–")} folds</th></tr></thead>
<tbody>{table}</tbody></table>
<p class="note">Cross-fitted means each image was scored by settings fitted without it. Ranges are 95% Clopper-Pearson intervals; with fewer than 100 images they are wide, so read the range, not just the number.</p>

<h2>Two ways of adapting, compared on the same cross-fitted images</h2>
<table><thead><tr><th>Strategy</th><th>ROC-AUC</th><th>Sensitivity</th><th>Specificity</th></tr></thead><tbody>{strat_rows}</tbody></table>
<p class="note">Recalibration cannot change how well the model ranks patients. A local adaptation layer can, so it is used only when it beats recalibration by a clear margin.</p>

<h2>Settings now in use</h2>
<div class="kv">
<div><b>{_num(dep["threshold"], 3)}</b><span>referral line, set for {_pct(dep["target_sensitivity"])} sensitivity</span></div>
<div><b>{escape(STRATEGY.get(dep.get("strategy", "recalibrate"), "Recalibrated"))}</b><span>adaptation strategy</span></div>
<div><b>{escape(q_text)}</b><span>confidence-check thresholds per class</span></div>
</div>

<h2>Warnings</h2><ul class="warn">{warns or "<li>None.</li>"}</ul>
<div class="sign"><div>Reviewed by (clinical lead)</div><div>Date</div></div>
</body></html>"""
