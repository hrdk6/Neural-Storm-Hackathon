"""Class-conditional (Mondrian) split conformal prediction for a binary screening task.

Marginal conformal guarantees can hide a coverage collapse on the rare positive class, and they weaken
under dataset shift. We therefore calibrate one quantile *per class* and, at each new site, recalibrate
on local labels and report the per-class coverage we actually observe - including when it fails.

With n_c calibration examples in class c, a finite quantile exists only if n_c >= ceil(1/alpha) - 1.
Below that the class can never be excluded, every set contains it, and the output is "Uncertain".
That is the honest behaviour for a site with too few labelled positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def min_per_class(alpha: float) -> int:
    return math.ceil(1 / alpha) - 1


@dataclass
class ClassConditionalConformal:
    alpha: float
    qhat: dict[int, float | None] = field(default_factory=dict)  # None = +inf (class always included)
    n: dict[int, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, p1: np.ndarray, y: np.ndarray, alpha: float) -> ClassConditionalConformal:
        p1 = np.asarray(p1, dtype=float)
        y = np.asarray(y, dtype=int)
        out = cls(alpha=alpha)
        for c in (0, 1):
            scores = np.sort(1.0 - np.where(c == 1, p1, 1 - p1)[y == c])
            n_c = len(scores)
            k = math.ceil((n_c + 1) * (1 - alpha))
            out.n[c] = n_c
            out.qhat[c] = float(scores[k - 1]) if 0 < k <= n_c else None
        return out

    def _q(self, c: int) -> float:
        q = self.qhat.get(c)
        return math.inf if q is None else q

    def predict_sets(self, p1: np.ndarray) -> np.ndarray:
        """-> bool array [n, 2]; column c says whether class c is in the prediction set."""
        p1 = np.atleast_1d(np.asarray(p1, dtype=float))
        return np.stack([(p1 <= self._q(0)), (1 - p1 <= self._q(1))], axis=1)

    def coverage(self, p1: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        sets = self.predict_sets(p1)
        y = np.asarray(y, dtype=int)
        covered = sets[np.arange(len(y)), y]
        out: dict[str, Any] = {"target": 1 - self.alpha, "overall": float(covered.mean()) if len(y) else None}
        for c in (0, 1):
            m = y == c
            out[f"class_{c}"] = float(covered[m].mean()) if m.any() else None
            out[f"n_{c}"] = int(m.sum())
        sizes = sets.sum(axis=1)
        out["singleton_rate"] = float((sizes == 1).mean()) if len(y) else None
        out["empty_rate"] = float((sizes == 0).mean()) if len(y) else None
        return out

    @property
    def is_informative(self) -> bool:
        return all(self.qhat.get(c) is not None for c in (0, 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "qhat": {str(k): v for k, v in self.qhat.items()},
            "n": {str(k): v for k, v in self.n.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClassConditionalConformal:
        return cls(
            alpha=float(d["alpha"]),
            qhat={int(k): (None if v is None else float(v)) for k, v in d["qhat"].items()},
            n={int(k): int(v) for k, v in d.get("n", {}).items()},
        )
