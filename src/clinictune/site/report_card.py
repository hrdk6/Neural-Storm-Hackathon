"""Printable HTML site report card (self-contained, no external assets)."""

from __future__ import annotations

from html import escape
from typing import Any

from clinictune import DISCLAIMER

VERDICT = {
    "ready": ("Ready for supervised use", "#1b7a43"),
    "use_with_caution": ("Use with caution", "#a15c00"),
    "not_ready": ("Not ready - do not rely on outputs", "#b42318"),
}


def _pct(v: Any) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _ci(ci: Any) -> str:
    return "" if not ci else f" <small>({ci[0] * 100:.0f}-{ci[1] * 100:.0f}%)</small>"


def _num(v: Any, d: int = 2) -> str:
    return "-" if v is None else f"{v:.{d}f}"


def render(report: dict[str, Any]) -> str:
    r = report
    label, colour = VERDICT.get(r["verdict"], (r["verdict"], "#444"))
    before = r["before_recalibration"]
    after = r["after_recalibration_crossfit"]
    bm, bc = before["metrics"], before["conformal_coverage"]
    am = after.get("metrics", {})
    ac = after.get("conformal_coverage", {})
    target = bc.get("target", 0.9)

    def cov_cell(v: Any) -> str:
        if v is None:
            return "-"
        ok = v >= target - 0.05
        return f'<span class="{"ok" if ok else "bad"}">{_pct(v)}</span>'

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
        ("Calibration error (ECE)", _num(bm.get("ece")), _num(am.get("ece"))),
        (
            f"Conformal coverage, negatives (target {target:.0%})",
            cov_cell(bc.get("class_0")),
            cov_cell(ac.get("class_0")),
        ),
        (
            f"Conformal coverage, positives (target {target:.0%})",
            cov_cell(bc.get("class_1")),
            cov_cell(ac.get("class_1")),
        ),
    ]
    table = "".join(f"<tr><th>{escape(a)}</th><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    shift = r["shift"]
    gate = r.get("quality_gate", {})
    warns = "".join(f"<li>{escape(w)}</li>" for w in r.get("warnings", []))
    reasons = "".join(f"<li>{escape(w)}</li>" for w in r.get("verdict_reasons", []))
    dep = r["deployed_calibration"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Site report card - {escape(r["site_id"])}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1d2433;max-width:860px;margin:24px auto;padding:0 16px}}
h1{{font-size:22px;margin:0}} h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid #dde2ea;padding-bottom:4px}}
.verdict{{display:inline-block;padding:6px 12px;border-radius:6px;color:#fff;background:{colour};font-weight:600;margin:12px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}}
td small{{color:#667085}} .ok{{color:#1b7a43;font-weight:600}} .bad{{color:#b42318;font-weight:600}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}}
.kv div{{background:#f6f8fb;border-radius:6px;padding:8px 10px}} .kv b{{display:block;font-size:18px}}
.disclaimer{{background:#fff7e6;border:1px solid #f5d18a;border-radius:6px;padding:10px 12px;font-size:13px}}
@media print{{body{{margin:0}}}}
</style></head><body>
<p class="disclaimer"><b>Screening aid / decision support only.</b> {escape(DISCLAIMER)}</p>
<h1>Site report card: {escape(r["site_id"])} &middot; {escape(r["task"].upper())}</h1>
<div>Calibration v{r.get("version", "?")} &middot; model bundle {escape(r["bundle_id"])} &middot; {escape(r["created_at"])}</div>
<div class="verdict">{escape(label)}</div>
<ul>{reasons}</ul>
<h2>Local data</h2>
<div class="kv">
<div><b>{r["counts"]["n"]}</b>labelled images used</div>
<div><b>{r["counts"]["n_pos"]} / {r["counts"]["n_neg"]}</b>positive / negative</div>
<div><b>{_pct(gate.get("pass_rate"))}</b>passed quality gate ({gate.get("passed", "-")}/{gate.get("submitted", "-")})</div>
</div>
<h2>Dataset shift vs. training sites</h2>
<div class="kv">
<div><b>{escape(str(shift.get("severity")))}</b>shift severity</div>
<div><b>{_num(shift.get("domain_auc"))}</b>domain-classifier AUC (0.5 = no shift)</div>
<div><b>{_num(shift.get("p_value"), 3)}</b>MMD permutation p-value</div>
<div><b>{_pct(shift.get("observed_prevalence"))}</b>local prevalence (training {_pct(shift.get("training_prevalence"))})</div>
</div>
<h2>Performance before vs. after site recalibration</h2>
<table><tr><th></th><th>Before (training-site calibration)</th><th>After (cross-fitted, {after.get("folds", "-")} folds)</th></tr>{table}</table>
<p><small>"After" numbers are cross-fitted: each image is scored by a calibration fitted without it. Intervals are
95% Clopper-Pearson. Small local samples give wide intervals - read the interval, not just the point estimate.</small></p>
<h2>Deployed settings</h2>
<div class="kv">
<div><b>{_num(dep["threshold"], 3)}</b>referral threshold (target sensitivity {_pct(dep["target_sensitivity"])})</div>
<div><b>{escape(str(dep["conformal"]["qhat"]))}</b>conformal quantiles per class (null = class never excluded)</div>
</div>
<h2>Warnings</h2><ul>{warns or "<li>None</li>"}</ul>
</body></html>"""
