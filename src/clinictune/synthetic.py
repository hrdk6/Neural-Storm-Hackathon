"""Synthetic chest X-rays and fundus photos for demos and tests.

These are cartoons with a controllable "lesion" signal and controllable per-site acquisition shift
(contrast, gamma, noise, blur, colour cast, prevalence) - plus an optional hospital *marker* that is
correlated with the label, to show that the shortcut test catches it. They are not medical images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from clinictune.data.manifest import Record


@dataclass(frozen=True)
class SiteProfile:
    name: str
    task: str
    n: int
    prevalence: float = 0.35
    contrast: float = 1.0
    gamma: float = 1.0
    noise: float = 6.0
    blur: float = 0.0
    lesion_strength: float = 1.0
    colour_cast: tuple[float, float, float] = (1.0, 1.0, 1.0)
    marker_corr: float = 0.0  # P(marker | positive) - P(marker | negative)


def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


def _blob(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))


def _finish(img: np.ndarray, p: SiteProfile, rng: np.random.Generator) -> np.ndarray:
    if p.blur:
        img = gaussian_filter(img, sigma=(p.blur, p.blur, 0) if img.ndim == 3 else p.blur)
    img = img + rng.normal(0, p.noise, img.shape)
    img = np.clip(img, 0, 255) / 255.0
    img = img**p.gamma
    img = np.clip((img - 0.5) * p.contrast + 0.5, 0, 1)
    return (img * 255).astype(np.uint8)


def cxr(label: int, p: SiteProfile, rng: np.random.Generator, size: int = 512) -> np.ndarray:
    h = w = size
    img = np.full((h, w), 18.0)
    img[_ellipse(h, w, h * 0.55, w * 0.5, h * 0.45, w * 0.44)] = 125
    lungs = _ellipse(h, w, h * 0.48, w * 0.29, h * 0.30, w * 0.15) | _ellipse(
        h, w, h * 0.48, w * 0.71, h * 0.30, w * 0.15
    )
    img[lungs] = 62
    img[_ellipse(h, w, h * 0.55, w * 0.5, h * 0.36, w * 0.07)] = 175
    img = gaussian_filter(img, 6)
    yy = np.arange(h)[:, None]
    img += lungs * 10 * np.sin(yy / (h * 0.035) + rng.uniform(0, 6)) ** 2
    if label:
        for _ in range(rng.integers(1, 4)):
            side = rng.choice([0.29, 0.71])
            cy = h * rng.uniform(0.25, 0.48)
            cx = w * (side + rng.uniform(-0.07, 0.07))
            img += 55 * p.lesion_strength * _blob(h, w, cy, cx, h * rng.uniform(0.025, 0.05))
    has_marker = rng.random() < (0.5 + p.marker_corr / 2 if label else 0.5 - p.marker_corr / 2)
    if p.marker_corr and has_marker:
        img[int(h * 0.04) : int(h * 0.10), int(w * 0.04) : int(w * 0.10)] = 250
    img = _finish(img, p, rng)
    return np.stack([img] * 3, axis=-1)


def fundus(label: int, p: SiteProfile, rng: np.random.Generator, size: int = 512) -> np.ndarray:
    h = w = size
    fov = _ellipse(h, w, h / 2, w / 2, h * 0.46, w * 0.46)
    r = np.hypot(*(np.mgrid[0:h, 0:w] - np.array([h / 2, w / 2])[:, None, None])) / (h * 0.46)
    vignette = np.clip(1.05 - 0.45 * r**2, 0, 1)
    img = np.zeros((h, w, 3))
    base = np.array([205.0, 95.0, 45.0]) * np.array(p.colour_cast)
    img[fov] = base
    img *= vignette[..., None]
    dy, dx = h * 0.5 + rng.uniform(-20, 20), w * rng.choice([0.32, 0.68])
    img += _blob(h, w, dy, dx, h * 0.05)[..., None] * np.array([45.0, 70.0, 40.0])
    for _ in range(8):  # vessels: random walks from the disc
        y, x = dy, dx
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(140):
            ang += rng.normal(0, 0.12)
            y, x = y + 2.2 * np.sin(ang), x + 2.2 * np.cos(ang)
            yi, xi = int(y), int(x)
            if 2 <= yi < h - 2 and 2 <= xi < w - 2:
                img[yi - 1 : yi + 2, xi - 1 : xi + 2] *= np.array([0.75, 0.55, 0.55])
    if label:
        for _ in range(rng.integers(12, 30)):  # haemorrhages / microaneurysms
            cy, cx = rng.uniform(0.2, 0.8, 2) * h
            img -= (
                _blob(h, w, cy, cx, rng.uniform(2.0, 4.5))[..., None]
                * np.array([90.0, 60.0, 30.0])
                * (p.lesion_strength)
            )
        for _ in range(rng.integers(3, 10)):  # hard exudates
            cy, cx = rng.uniform(0.25, 0.75, 2) * h
            img += (
                _blob(h, w, cy, cx, rng.uniform(2.0, 3.5))[..., None]
                * np.array([40.0, 90.0, 10.0])
                * (p.lesion_strength)
            )
    img = gaussian_filter(img, sigma=(1.0, 1.0, 0))
    img[~fov] = 0
    out = _finish(img, p, rng)
    out[~fov] = rng.integers(0, 6, (int((~fov).sum()), 3))
    return out


GENERATORS = {"tb": cxr, "dr": fundus}


def generate_site(p: SiteProfile, out_dir: str | Path, seed: int = 0) -> list[Record]:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir) / p.name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(p.n):
        label = int(rng.random() < p.prevalence)
        img = GENERATORS[p.task](label, p, rng)
        path = out_dir / f"{p.name}_{i:04d}.png"
        Image.fromarray(img).save(path)
        records.append(
            Record(
                image_path=str(path.resolve()),
                label=label,
                patient_id=f"{p.name}:{i // 2}" if p.task == "dr" else f"{p.name}:{i}",
                site=p.name,
                task=p.task,
                dataset="synthetic",
            )
        )
    return records


TRAIN_SITES = [
    SiteProfile("tb_site_a", "tb", 360, prevalence=0.35),
    SiteProfile("tb_site_b", "tb", 360, prevalence=0.25, contrast=0.85, noise=10),
    SiteProfile("tb_site_c", "tb", 360, prevalence=0.40, gamma=1.2, marker_corr=0.6),
    SiteProfile("dr_site_a", "dr", 360, prevalence=0.30),
    SiteProfile("dr_site_b", "dr", 360, prevalence=0.25, colour_cast=(0.95, 1.05, 1.1), blur=0.8),
]
NEW_CLINICS = [
    SiteProfile(
        "tb_clinic_new", "tb", 160, prevalence=0.30, gamma=0.8, contrast=0.85, noise=10, blur=0.6, lesion_strength=0.95
    ),
    SiteProfile(
        "dr_clinic_new",
        "dr",
        160,
        prevalence=0.35,
        colour_cast=(1.1, 0.9, 0.85),
        noise=9,
        blur=1.0,
        lesion_strength=0.8,
    ),
]
