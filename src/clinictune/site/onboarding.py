"""Site onboarding: a clinic uploads 50-100 locally labelled images and gets

1. a dataset-shift measurement against the training sites,
2. *before* metrics - the training-site model on local data, including conformal coverage per class
   (this is where coverage typically fails, and we say so),
3. *after* metrics for two adaptation strategies, both cross-fitted so no image is scored by anything
   fitted on its own label:
     - ``recalibrate``: MAP Platt + sensitivity-targeted threshold + class-conditional conformal,
     - ``adapt``: a regularised local residual head (:mod:`clinictune.model.adapt`) + the same calibration
       fitted on out-of-fold logits - the only option that can improve *ranking* at a new site,
4. the better strategy deployed (adapt must beat recalibration by a margin, so it is not chosen on noise),
5. a verdict with the reasons behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from clinictune.model.adapt import ResidualHead, select_and_fit
from clinictune.model.bundle import TaskModel
from clinictune.model.calibration import Platt, fit_platt, threshold_for_sensitivity
from clinictune.model.conformal import ClassConditionalConformal, min_per_class
from clinictune.model.metrics import binary_report, check_targets
from clinictune.model.shift import bbse_prevalence, shift_report

MIN_TOTAL = 30
RECOMMENDED_TOTAL = 100
ADAPT_MIN_PER_CLASS = 10
ADAPT_MARGIN = 0.02  # cross-fitted ROC-AUC gain required before deploying a local residual head


@dataclass
class SiteCalibration:
    site_id: str
    task: str
    bundle_id: str
    platt: Platt
    threshold: float
    conformal: ClassConditionalConformal
    ood_threshold: float
    residual: ResidualHead | None = None
    strategy: str = "recalibrate"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    version: int = 0

    def logit(self, tm: TaskModel, emb: np.ndarray) -> np.ndarray:
        z = tm.head.logit(emb)
        return z + self.residual.logit(tm.head, emb) if self.residual is not None else z

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "task": self.task,
            "bundle_id": self.bundle_id,
            "platt": self.platt.to_dict(),
            "threshold": self.threshold,
            "conformal": self.conformal.to_dict(),
            "ood_threshold": self.ood_threshold,
            "residual": self.residual.to_dict() if self.residual else None,
            "strategy": self.strategy,
            "created_at": self.created_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SiteCalibration:
        return cls(
            site_id=d["site_id"],
            task=d["task"],
            bundle_id=d["bundle_id"],
            platt=Platt.from_dict(d["platt"]),
            threshold=float(d["threshold"]),
            conformal=ClassConditionalConformal.from_dict(d["conformal"]),
            ood_threshold=float(d["ood_threshold"]),
            residual=ResidualHead.from_dict(d["residual"]) if d.get("residual") else None,
            strategy=d.get("strategy", "recalibrate"),
            created_at=d.get("created_at", ""),
            version=int(d.get("version", 0)),
        )


def _calibrate(
    tm: TaskModel, logit: np.ndarray, y: np.ndarray, alpha: float, prior_strength: float
) -> tuple[Platt, float, ClassConditionalConformal]:
    platt = fit_platt(logit, y, prior=tm.platt, prior_logits=tm.head.logit(tm.reference), prior_strength=prior_strength)
    p = platt(logit)
    threshold = threshold_for_sensitivity(p, y, tm.target_sensitivity) if y.sum() else tm.threshold
    return platt, threshold, ClassConditionalConformal.fit(p, y, alpha)


# Backwards-compatible name used by the LOSO experiments.
_fit_site = _calibrate

FoldFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray] | None]


def _cross_fit(y: np.ndarray, groups: np.ndarray, fold_fn: FoldFn, alpha: float, seed: int) -> dict[str, Any]:
    """Generic cross-fitting: ``fold_fn(train_idx, test_idx)`` -> (p_test, pred_test, sets_test)."""
    n_splits = int(min(5, np.bincount(y, minlength=2).min(), len(np.unique(groups))))
    if n_splits < 2:
        return {"available": False, "reason": "need at least 2 labelled images of each class"}
    p_out = np.zeros(len(y))
    pred_out = np.zeros(len(y), dtype=bool)
    covered = np.zeros(len(y), dtype=bool)
    set_size = np.zeros(len(y), dtype=int)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        out = fold_fn(tr, te)
        if out is None:
            return {"available": False, "reason": "too few labels per class inside a fold"}
        p, pred, sets = out
        p_out[te], pred_out[te] = p, pred
        covered[te] = sets[np.arange(len(te)), y[te]]
        set_size[te] = sets.sum(axis=1)
    coverage: dict[str, Any] = {"target": 1 - alpha, "overall": float(covered.mean())}
    for c in (0, 1):
        m = y == c
        coverage[f"class_{c}"] = float(covered[m].mean()) if m.any() else None
        coverage[f"n_{c}"] = int(m.sum())
    coverage["singleton_rate"] = float((set_size == 1).mean())
    return {
        "available": True,
        "folds": n_splits,
        "metrics": binary_report(y, p_out, float("nan"), groups, pred=pred_out),
        "conformal_coverage": coverage,
    }


def onboard_site(
    tm: TaskModel,
    site_id: str,
    bundle_id: str,
    emb: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    alpha: float | None = None,
    prior_strength: float = 5.0,
    gate_summary: dict[str, Any] | None = None,
    allow_adapt: bool = True,
    seed: int = 0,
) -> tuple[SiteCalibration, dict[str, Any]]:
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups) if groups is not None else np.arange(len(y)).astype(str)
    alpha = alpha if alpha is not None else tm.conformal.alpha
    if len(y) < MIN_TOTAL:
        raise ValueError(f"onboarding needs at least {MIN_TOTAL} labelled images that pass the quality gate")

    logit = tm.head.logit(emb)
    p_src = tm.platt(logit)
    counts = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}

    shift = shift_report(tm.reference, emb, seed=seed)
    src_metrics = tm.metrics.get("overall", {})
    if src_metrics.get("sensitivity") is not None and src_metrics.get("specificity") is not None:
        shift["bbse_prevalence_estimate"] = bbse_prevalence(
            p_src, tm.threshold, src_metrics["sensitivity"], src_metrics["specificity"]
        )
    shift["observed_prevalence"] = float(y.mean())
    shift["training_prevalence"] = src_metrics.get("prevalence")

    before = {
        "metrics": binary_report(y, p_src, tm.threshold, groups),
        "conformal_coverage": tm.conformal.coverage(p_src, y),
    }

    # --- strategy 1: recalibrate the training-site head
    def recal_fold(tr: np.ndarray, te: np.ndarray):  # type: ignore[no-untyped-def]
        platt, thr, conf = _calibrate(tm, logit[tr], y[tr], alpha, prior_strength)
        p = platt(logit[te])
        return p, p >= thr, conf.predict_sets(p)

    strategies: dict[str, dict[str, Any]] = {"recalibrate": _cross_fit(y, groups, recal_fold, alpha, seed)}

    # --- strategy 2: local residual head, calibrated on out-of-fold logits
    enough = min(counts["n_pos"], counts["n_neg"]) >= ADAPT_MIN_PER_CLASS
    if allow_adapt and enough:

        def adapt_fold(tr: np.ndarray, te: np.ndarray):  # type: ignore[no-untyped-def]
            sel = select_and_fit(tm.head, emb[tr], y[tr], groups[tr], seed)
            if sel is None:
                return None
            residual, oof, _ = sel
            platt, thr, conf = _calibrate(tm, oof, y[tr], alpha, 0.0)
            p = platt(logit[te] + residual.logit(tm.head, emb[te]))
            return p, p >= thr, conf.predict_sets(p)

        strategies["adapt"] = _cross_fit(y, groups, adapt_fold, alpha, seed)
    else:
        strategies["adapt"] = {
            "available": False,
            "reason": f"needs >= {ADAPT_MIN_PER_CLASS} labelled images of each class" if allow_adapt else "disabled",
        }

    chosen, why = _choose(strategies)
    residual = None
    if chosen == "adapt":
        sel = select_and_fit(tm.head, emb, y, groups, seed)
        assert sel is not None
        residual, oof, lam_auc = sel
        platt, threshold, conformal = _calibrate(tm, oof, y, alpha, 0.0)
        strategies["adapt"]["inner_cv_auc_by_lambda"] = lam_auc
        strategies["adapt"]["lambda"] = residual.lam
    else:
        platt, threshold, conformal = _calibrate(tm, logit, y, alpha, prior_strength)
    after = strategies[chosen]

    ood_scores = tm.ood.score(emb)
    ood_threshold = float(max(tm.ood.threshold, np.percentile(ood_scores, 99)))
    calib = SiteCalibration(
        site_id=site_id,
        task=tm.task,
        bundle_id=bundle_id,
        platt=platt,
        threshold=threshold,
        conformal=conformal,
        ood_threshold=ood_threshold,
        residual=residual,
        strategy=chosen,
    )

    warnings: list[str] = []
    need = min_per_class(alpha)
    for c, name in ((1, "positive"), (0, "negative")):
        n_c = int((y == c).sum())
        if n_c < need:
            warnings.append(
                f"Only {n_c} {name} images: at least {need} are needed for a {int((1 - alpha) * 100)}% "
                f"class-conditional guarantee. The model will answer 'Uncertain' instead of excluding this class."
            )
    if counts["n_pos"] < 20:
        warnings.append(
            f"The referral threshold was set from {counts['n_pos']} positive images; its sensitivity estimate "
            "is imprecise (see the confidence interval). Label more positives when possible."
        )
    if len(y) < RECOMMENDED_TOTAL:
        warnings.append(f"{len(y)} images labelled; {RECOMMENDED_TOTAL} are recommended.")
    cov_before = before["conformal_coverage"]
    for c in (0, 1):
        v = cov_before.get(f"class_{c}")
        if v is not None and v < (1 - alpha) - 0.05:
            warnings.append(
                f"Before recalibration, coverage for class {c} was {v:.0%} (target {1 - alpha:.0%}): the "
                "training-site guarantee did not transfer to this site."
            )

    verdict, reasons = _verdict(tm.task, after, conformal, shift)
    report = {
        "site_id": site_id,
        "task": tm.task,
        "bundle_id": bundle_id,
        "created_at": calib.created_at,
        "counts": counts,
        "quality_gate": gate_summary or {},
        "shift": shift,
        "before_recalibration": before,
        "strategies": strategies,
        "strategy": chosen,
        "strategy_reason": why,
        "after_recalibration_crossfit": after,
        "deployed_calibration": {
            "strategy": chosen,
            "platt": platt.to_dict(),
            "threshold": threshold,
            "target_sensitivity": tm.target_sensitivity,
            "conformal": conformal.to_dict(),
            "ood_threshold": ood_threshold,
            "prior_strength": prior_strength if chosen == "recalibrate" else 0.0,
            "residual_lambda": residual.lam if residual else None,
        },
        "targets": check_targets(tm.task, after["metrics"]) if after.get("available") else None,
        "warnings": warnings,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }
    return calib, report


def _choose(strategies: dict[str, dict[str, Any]]) -> tuple[str, str]:
    r, a = strategies["recalibrate"], strategies["adapt"]
    if not a.get("available"):
        return "recalibrate", f"Local adaptation not tried: {a.get('reason')}."
    if not r.get("available"):
        return "adapt", "Recalibration could not be evaluated."
    ra, aa = r["metrics"].get("roc_auc") or 0, a["metrics"].get("roc_auc") or 0
    if aa >= ra + ADAPT_MARGIN:
        return "adapt", (
            f"A local adaptation layer raised cross-fitted ROC-AUC from {ra:.3f} to {aa:.3f} "
            f"(needs >= +{ADAPT_MARGIN:.2f}); it is deployed with calibration fitted on out-of-fold scores."
        )
    return "recalibrate", (
        f"Recalibration kept: a local adaptation layer reached ROC-AUC {aa:.3f} vs {ra:.3f}, "
        f"below the +{ADAPT_MARGIN:.2f} margin needed to justify a site-specific model."
    )


def _verdict(
    task: str, after: dict[str, Any], conformal: ClassConditionalConformal, shift: dict[str, Any]
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not after.get("available"):
        return "not_ready", [after.get("reason", "cross-fitted evaluation unavailable")]
    m = after["metrics"]
    t = check_targets(task, m)
    if not conformal.is_informative:
        reasons.append("Too few labels in at least one class for an informative conformal layer.")
    sens_lo = m["sensitivity_ci"][0]
    if sens_lo < t["targets"]["sensitivity"] - 0.25:
        reasons.append(f"Sensitivity lower confidence bound {sens_lo:.0%} is far below target.")
    spec_lo = m["specificity_ci"][0]
    if spec_lo < t["targets"]["specificity"] - 0.35:
        reasons.append(
            f"Specificity lower confidence bound {spec_lo:.0%} is far below target: most healthy "
            "patients would be referred."
        )
    if m.get("roc_auc") is not None and m["roc_auc"] < 0.70:
        reasons.append(
            f"ROC-AUC {m['roc_auc']:.2f}: the model barely separates the classes at this site, even after "
            "local adaptation. A different backbone or more local labels are needed."
        )
    if reasons:
        return "not_ready", reasons
    if not t["point_estimate_meets"]:
        for key, name in (("sensitivity", "Sensitivity"), ("specificity", "Specificity")):
            value, target = m.get(key) or 0.0, t["targets"][key]
            if value < target:
                reasons.append(f"{name} {value:.1%} is below the {target:.0%} screening target at this site.")
    cov = after["conformal_coverage"]
    for c in (0, 1):
        v = cov.get(f"class_{c}")
        if v is not None and v < cov["target"] - 0.05:
            reasons.append(f"Cross-fitted coverage for class {c} is {v:.0%} (target {cov['target']:.0%}).")
    if shift.get("severity") == "severe":
        reasons.append("Severe dataset shift versus the training sites.")
    if reasons:
        return "use_with_caution", reasons
    return "ready", ["Cross-fitted metrics meet targets and per-class coverage holds."]
