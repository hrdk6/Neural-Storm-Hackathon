"""Probability calibration and operating thresholds.

Site recalibration is MAP Platt scaling: the training-site calibration enters as a few soft-labelled
pseudo-observations, which keeps the fit stable with 10 local labels and lets the data take over by ~100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import beta


@dataclass
class Platt:
    a: float = 1.0
    b: float = 0.0

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        return expit(self.a * np.asarray(logit, dtype=float) + self.b)

    def to_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Platt:
        return cls(a=float(d["a"]), b=float(d["b"]))


def fit_platt(
    logit: np.ndarray,
    y: np.ndarray,
    prior: Platt | None = None,
    prior_logits: np.ndarray | None = None,
    prior_strength: float = 0.0,
) -> Platt:
    """Platt scaling, optionally MAP with a pseudo-observation prior.

    The prior is ``prior_strength`` soft-labelled pseudo-images: the training-site logits ``prior_logits``
    labelled with the training-site calibration ``prior``. So "prior_strength=5" literally means the old
    calibration counts as 5 labelled images - independent of the logit scale.
    """
    z = np.asarray(logit, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = prior or Platt()
    use_prior = prior_strength > 0 and prior_logits is not None and len(prior_logits) > 0
    zr = np.asarray(prior_logits, dtype=float) if use_prior else np.empty(0)
    yr = p0(zr) if use_prior else np.empty(0)
    wr = prior_strength / max(len(zr), 1)
    # Optimise on standardised logits for conditioning; map back at the end.
    scale = float(np.std(np.r_[z, zr])) or 1.0
    zs, zrs = z / scale, zr / scale

    def loss(theta: np.ndarray) -> float:
        a, b = theta
        s = a * zs + b
        nll = -np.sum(y * log_expit(s) + (1 - y) * log_expit(-s))
        if use_prior:
            sr = a * zrs + b
            nll -= wr * np.sum(yr * log_expit(sr) + (1 - yr) * log_expit(-sr))
        return float(nll)

    x0 = np.array([p0.a * scale, p0.b])
    x0[0] = np.clip(x0[0], 1e-3, 100.0)
    x0[1] = np.clip(x0[1], -100.0, 100.0)
    res = minimize(loss, x0=x0, method="L-BFGS-B", bounds=[(1e-3, 100.0), (-100.0, 100.0)])
    return Platt(a=float(res.x[0] / scale), b=float(res.x[1]))


def threshold_for_sensitivity(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Threshold meeting the target sensitivity on (p, y), placed mid-gap for margin.

    The highest threshold that keeps empirical sensitivity >= target sits exactly on a positive's score, so
    any slightly weaker unseen positive falls below it. Moving down to the midpoint between that positive and
    the next negative below keeps the same empirical sensitivity *and* specificity on (p, y) while leaving
    margin for new positives - which matters most with the 5-20 positives a clinic labels.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))  # number of positives we may miss
    t = float(pos[min(k, len(pos) - 1)])
    below = p[(y == 0) & (p < t)]
    return (t + float(below.max())) / 2 if below.size else t


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
