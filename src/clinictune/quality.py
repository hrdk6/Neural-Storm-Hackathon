"""Image quality / gradability gate. Runs before the model; a failed gate means "Uncertain - retake".

Checks are deliberately simple, explainable image statistics with conservative thresholds. They catch the
common failures in low-resource capture (phone photos of films, blur, over/under-exposure, wrong modality,
lateral CXRs, fundus photos without a visible field of view). The embedding-space OOD check in the
triage step is the second line of defence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import laplace, uniform_filter

from clinictune.imaging import to_gray


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    limit: str
    message: str


@dataclass
class GateResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


def _work(gray: np.ndarray, size: int = 512) -> np.ndarray:
    """Downsample by striding to ~size px on the long side (stats are scale-robust; keeps the gate fast)."""
    step = max(1, max(gray.shape) // size)
    return gray[::step, ::step]


def _sharpness(gray: np.ndarray) -> float:
    g = gray / 255.0
    return float(laplace(g).var() / max(g.var(), 1e-6))


def _common(rgb: np.ndarray, gray: np.ndarray, min_side: int, min_contrast: float) -> list[Check]:
    h, w = rgb.shape[:2]
    clipped = float(np.mean((gray <= 2) | (gray >= 253)))
    return [
        Check(
            "resolution",
            min(h, w) >= min_side,
            float(min(h, w)),
            f">= {min_side}px",
            "Image resolution is too low. Use the original file, not a thumbnail or screenshot.",
        ),
        Check(
            "contrast",
            float(gray.std()) >= min_contrast,
            float(gray.std()),
            f">= {min_contrast}",
            "Image is washed out or almost uniform.",
        ),
    ] + (
        []
        if clipped < 0.5
        else [
            Check("exposure", False, clipped, "< 50% clipped pixels", "Image is badly over- or under-exposed."),
        ]
    )


def check_cxr(rgb: np.ndarray) -> GateResult:
    gray = _work(to_gray(rgb))
    checks = _common(rgb, gray, 384, 20.0)
    chroma = float(np.abs(rgb[..., 0].astype(int) - rgb[..., 2].astype(int)).mean())
    checks.append(
        Check(
            "greyscale",
            chroma < 12,
            chroma,
            "< 12",
            "Image is in colour - this does not look like a chest X-ray (or is a photo of a screen).",
        )
    )
    h, w = rgb.shape[:2]
    aspect = w / h
    checks.append(
        Check(
            "aspect_ratio",
            0.6 <= aspect <= 1.6,
            aspect,
            "0.6-1.6",
            "Unusual aspect ratio - is the image cropped, rotated or a lateral view?",
        )
    )
    sharp = _sharpness(gray)
    checks.append(Check("sharpness", sharp >= 0.002, sharp, ">= 0.002", "Image looks blurred."))

    # Lung fields: in a frontal CXR the two lung zones (mid-left, mid-right) are darker than the mediastinum.
    hh, ww = gray.shape
    band = gray[int(hh * 0.25) : int(hh * 0.70)]
    left = band[:, int(ww * 0.15) : int(ww * 0.40)].mean()
    mid = band[:, int(ww * 0.43) : int(ww * 0.57)].mean()
    right = band[:, int(ww * 0.60) : int(ww * 0.85)].mean()
    lung_contrast = float((mid - (left + right) / 2) / (gray.std() + 1e-6))
    checks.append(
        Check(
            "lungs_visible",
            lung_contrast >= 0.15,
            lung_contrast,
            ">= 0.15",
            "Could not find two lung fields either side of the mediastinum. Check it is a frontal "
            "(PA/AP) chest X-ray, not inverted, and that both lungs are fully in view.",
        )
    )
    return GateResult(all(c.passed for c in checks), checks)


def check_fundus(rgb: np.ndarray) -> GateResult:
    small = rgb[:: max(1, max(rgb.shape[:2]) // 512), :: max(1, max(rgb.shape[:2]) // 512)].astype(np.float32)
    gray = to_gray(small)
    fov = gray > 20
    fov_frac = float(fov.mean())
    checks = _common(rgb, gray[fov] if fov.any() else gray, 512, 8.0)
    checks.append(
        Check(
            "field_of_view",
            0.35 <= fov_frac <= 0.97,
            fov_frac,
            "0.35-0.97",
            "No circular retinal field of view found - is this a fundus photograph?",
        )
    )
    if fov.any():
        r, g, b = (small[..., i][fov].mean() for i in range(3))
        red_ratio = float(r / (g + b + 1e-6))
        checks.append(
            Check(
                "colour",
                red_ratio >= 0.8,
                red_ratio,
                ">= 0.8",
                "Colours do not look like a retina (expected red/orange).",
            )
        )
        local_mean = uniform_filter(gray, size=31)
        inside = local_mean[fov]
        uniformity = float(np.percentile(inside, 10) / (np.percentile(inside, 90) + 1e-6))
        checks.append(
            Check(
                "illumination",
                uniformity >= 0.25,
                uniformity,
                ">= 0.25",
                "Uneven illumination - part of the retina is too dark. Retake with better focus/pupil.",
            )
        )
        sharp = _sharpness(np.where(fov, gray, gray[fov].mean()))
        checks.append(Check("sharpness", sharp >= 0.001, sharp, ">= 0.001", "Image looks blurred."))
    return GateResult(all(c.passed for c in checks), checks)


GATES = {"tb": check_cxr, "dr": check_fundus}


def check(task: str, rgb: np.ndarray) -> GateResult:
    return GATES[task](rgb)
