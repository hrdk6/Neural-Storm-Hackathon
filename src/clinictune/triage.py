"""Per-image screening: gate -> embedding -> OOD -> calibrated probability -> conformal set -> decision.

Decision rule (sensitivity-first; missing disease is the costlier error in screening):

    gate failed (rules or learned gradability)    -> UNCERTAIN (retake)
    embedding out of distribution                 -> UNCERTAIN (escalate)
    set == {refer}                                -> REFER
    set == {refer, low} and p >= site threshold   -> REFER
    set == {low}        and p <  site threshold   -> LOW_RISK
    anything else (ambiguous, empty, conflicting) -> UNCERTAIN (escalate)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from clinictune import DISCLAIMER, explain, i18n, imaging, quality
from clinictune.model.adapt import ResidualHead
from clinictune.model.bundle import Bundle
from clinictune.model.calibration import Platt
from clinictune.model.conformal import ClassConditionalConformal
from clinictune.quality import Check
from clinictune.site.onboarding import SiteCalibration


class Decision(StrEnum):
    REFER = "refer"
    LOW_RISK = "low_risk"
    UNCERTAIN = "uncertain"


@dataclass
class Calibration:
    platt: Platt
    threshold: float
    conformal: ClassConditionalConformal
    ood_threshold: float
    source: str  # "site:<id>:v<n>" or "training-sites"
    residual: ResidualHead | None = None
    strategy: str = "training-sites"


def decide(p: float, in_set: np.ndarray, threshold: float) -> tuple[Decision, str]:
    low, refer = bool(in_set[0]), bool(in_set[1])
    if refer and not low:
        return Decision.REFER, "conformal_refer"
    if refer and low:
        if p >= threshold:
            return Decision.REFER, "ambiguous_set_above_threshold"
        return Decision.UNCERTAIN, "ambiguous_set"
    if low and not refer:
        if p < threshold:
            return Decision.LOW_RISK, "conformal_low_risk"
        return Decision.UNCERTAIN, "conflict_threshold_vs_conformal"
    return Decision.UNCERTAIN, "empty_set"


class Screener:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle

    def calibration_for(self, task: str, site: SiteCalibration | None) -> Calibration:
        tm = self.bundle.tasks[task]
        if site is not None and site.bundle_id == self.bundle.bundle_id:
            return Calibration(
                site.platt,
                site.threshold,
                site.conformal,
                site.ood_threshold,
                f"site:{site.site_id}:v{site.version}",
                site.residual,
                site.strategy,
            )
        return Calibration(tm.platt, tm.threshold, tm.conformal, tm.ood.threshold, "training-sites")

    def screen(
        self,
        data: bytes,
        task: str,
        *,
        site: SiteCalibration | None = None,
        lang: str = "en",
        heatmap: bool = True,
        similar: int = 6,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if task not in self.bundle.tasks:
            raise KeyError(f"task {task!r} not in bundle (has {sorted(self.bundle.tasks)})")
        tm = self.bundle.tasks[task]
        backbone = self.bundle.backbone_for(task)
        cal = self.calibration_for(task, site)
        rgb = imaging.decode(data)
        gate = quality.check(task, rgb)
        result: dict[str, Any] = {
            "task": task,
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "calibration": cal.source,
            "adaptation": cal.strategy,
            "model": {"bundle_id": self.bundle.bundle_id, "backbone": backbone.spec.name, "demo": self.bundle.is_demo},
            "disclaimer": DISCLAIMER,
            "warnings": [],
        }
        if cal.source == "training-sites":
            result["warnings"].append(
                "This site has not been onboarded: thresholds come from the training sites and may not hold here."
            )
        if self.bundle.is_demo:
            result["warnings"].append("DEMO model trained on synthetic images - outputs are meaningless clinically.")

        timings: dict[str, float] = {}
        emb = tokens = None
        model_rgb, view_region = rgb, None
        if gate.passed:
            t1 = time.perf_counter()
            model_rgb, view_region = imaging.model_view(tm.preprocess, rgb)
            emb, tokens = backbone.embed([model_rgb])
            timings["backbone_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            if tm.gradability is not None:
                p_bad = float(tm.gradability.p_ungradable(emb)[0])
                ok = p_bad < tm.gradability.threshold
                gate.checks.append(
                    Check(
                        "learned_gradability",
                        ok,
                        p_bad,
                        f"< {tm.gradability.threshold:.2f}",
                        "A model trained on expert image-quality labels judges this image ungradable "
                        "(focus, exposure, artefacts or field of view). Please retake.",
                    )
                )
                gate.passed = gate.passed and ok
        result["quality_gate"] = gate.to_dict()

        if not gate.passed:
            result.update(
                decision=Decision.UNCERTAIN,
                reason="quality_gate",
                probability=None,
                prediction_set=None,
                ood=None,
                note=i18n.note(lang, "retake", task),
            )
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return result

        assert emb is not None and tokens is not None
        ood_score = float(tm.ood.score(emb)[0])
        logit = tm.head.logit(emb)
        if cal.residual is not None:
            logit = logit + cal.residual.logit(tm.head, emb)
        p = float(cal.platt(logit)[0])
        in_set = cal.conformal.predict_sets(np.array([p]))[0]
        labels = [name for name, keep in zip(("low_risk", "refer"), in_set, strict=True) if keep]
        ood = {"score": ood_score, "threshold": cal.ood_threshold, "flagged": ood_score > cal.ood_threshold}
        if ood["flagged"]:
            decision, reason = Decision.UNCERTAIN, "out_of_distribution"
        else:
            decision, reason = decide(p, in_set, cal.threshold)
        note_key = {Decision.REFER: "refer", Decision.LOW_RISK: "low_risk", Decision.UNCERTAIN: "uncertain"}
        result.update(
            decision=decision,
            reason=reason,
            probability=p,
            threshold=cal.threshold,
            prediction_set=labels,
            ood=ood,
            note=i18n.note(lang, note_key[decision], task),
        )
        if heatmap:
            contrib = tm.head.token_contributions(tokens[0])
            if cal.residual is not None:
                contrib = contrib + cal.residual.token_contributions(tm.head, tokens[0])
            grid = contrib.reshape(backbone.spec.grid)
            result["heatmap_png"] = explain.to_data_uri(explain.overlay_png(rgb, grid, region=view_region))
            regions = explain.top_regions(grid, k=3, valid=explain.content_mask(model_rgb, backbone.spec.grid))
            if view_region is not None:
                regions = explain.regions_to_original(regions, view_region, rgb.shape)
            result["regions"] = regions
        if similar and tm.library is not None:
            result["similar_cases"] = tm.library.nearest(emb[0], k=similar)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["timings"] = timings
        result["latency_ms"] = timings["total_ms"]
        return result
