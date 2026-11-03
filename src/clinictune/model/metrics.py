"""Per-site screening metrics with uncertainty. Accuracy alone is never reported."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from clinictune.model.calibration import clopper_pearson, expected_calibration_error


def _safe(fn, y: np.ndarray, s: np.ndarray) -> float | None:  # type: ignore[no-untyped-def]
    return float(fn(y, s)) if len(np.unique(y)) == 2 else None


def binary_report(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    groups: np.ndarray | None = None,
    n_boot: int = 300,
    seed: int = 0,
    pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sensitivity / specificity (Clopper-Pearson CIs), ROC-AUC and PR-AUC (patient-bootstrap CIs), ECE.

    ``pred`` overrides ``p >= threshold`` (used for cross-fitted reports where each fold has its own threshold).
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = np.asarray(pred, dtype=bool) if pred is not None else p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    out: dict[str, Any] = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "threshold": None if np.isnan(threshold) else float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "sensitivity_ci": clopper_pearson(tp, tp + fn),
        "specificity": tn / (tn + fp) if tn + fp else None,
        "specificity_ci": clopper_pearson(tn, tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else None,
        "npv": tn / (tn + fn) if tn + fn else None,
        "roc_auc": _safe(roc_auc_score, y, p),
        "pr_auc": _safe(average_precision_score, y, p),
        "ece": expected_calibration_error(p, y) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
    }
    if n_boot and len(np.unique(y)) == 2:
        rng = np.random.default_rng(seed)
        g = np.asarray(groups) if groups is not None else np.arange(len(y))
        uniq = np.unique(g)
        members = {u: np.flatnonzero(g == u) for u in uniq}
        rocs, prs = [], []
        for _ in range(n_boot):
            idx = np.concatenate([members[u] for u in rng.choice(uniq, len(uniq), replace=True)])
            if len(np.unique(y[idx])) < 2:
                continue
            rocs.append(roc_auc_score(y[idx], p[idx]))
            prs.append(average_precision_score(y[idx], p[idx]))
        if rocs:
            out["roc_auc_ci"] = (float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5)))
            out["pr_auc_ci"] = (float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5)))
    return out


# WHO target product profile for a TB *triage* test (2014): sensitivity >= 0.90, specificity >= 0.70.
# For DR, commonly cited screening targets (e.g. UK NSC / FDA-cleared autonomous AI endpoints) are around
# sensitivity >= 0.85 and specificity >= 0.80. These are reference points, not regulatory claims.
TARGETS: dict[str, dict[str, float]] = {
    "tb": {"sensitivity": 0.90, "specificity": 0.70},
    "dr": {"sensitivity": 0.85, "specificity": 0.80},
}


def check_targets(task: str, report: dict[str, Any]) -> dict[str, Any]:
    t = TARGETS[task]
    sens_lo = report["sensitivity_ci"][0]
    spec_lo = report["specificity_ci"][0]
    return {
        "targets": t,
        "point_estimate_meets": bool(
            (report["sensitivity"] or 0) >= t["sensitivity"] and (report["specificity"] or 0) >= t["specificity"]
        ),
        "ci_lower_bound_meets": bool(sens_lo >= t["sensitivity"] and spec_lo >= t["specificity"]),
    }
