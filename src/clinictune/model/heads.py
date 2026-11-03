"""Linear heads on frozen, mean-pooled patch embeddings.

Because the image embedding is the *mean* of the patch tokens and the head is linear, the logit decomposes
exactly into per-patch contributions:  logit = mean_t(w . (tok_t - mu) / sigma) + b.
That gives a faithful heatmap from a single forward pass (see :mod:`clinictune.explain`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def logit(self, emb: np.ndarray) -> np.ndarray:
        return ((emb - self.mean) / self.scale) @ self.coef + self.intercept

    def token_contributions(self, tokens: np.ndarray) -> np.ndarray:
        """tokens [N, D] -> per-token contribution [N]; their mean + intercept equals the image logit."""
        return ((tokens - self.mean) / self.scale) @ self.coef

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LinearHead:
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            scale=np.asarray(d["scale"], dtype=np.float32),
            coef=np.asarray(d["coef"], dtype=np.float32),
            intercept=float(d["intercept"]),
        )


def _fit(X: np.ndarray, y: np.ndarray, C: float, sample_weight: np.ndarray | None, seed: int) -> LinearHead:
    scaler = StandardScaler().fit(X, sample_weight=sample_weight)
    scale = np.where(scaler.scale_ > 1e-8, scaler.scale_, 1.0)
    Z = (X - scaler.mean_) / scale
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
    lr.fit(Z, y, sample_weight=sample_weight)
    return LinearHead(
        mean=scaler.mean_.astype(np.float32),
        scale=scale.astype(np.float32),
        coef=lr.coef_[0].astype(np.float32),
        intercept=float(lr.intercept_[0]),
    )


def fit_head(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    Cs: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
    sample_weight: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[LinearHead, dict[str, Any]]:
    """Pick C by patient-grouped CV ROC-AUC on the training data only, then refit on all of it."""
    n_splits = int(min(5, np.bincount(y).min(), len(np.unique(groups))))
    if n_splits < 2:
        C = 0.1
        return _fit(X, y, C, sample_weight, seed), {"C": C, "cv": "skipped (too few samples)"}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: dict[float, list[float]] = {C: [] for C in Cs}
    for tr, va in cv.split(X, y, groups):
        if len(np.unique(y[va])) < 2:
            continue
        sw = sample_weight[tr] if sample_weight is not None else None
        for C in Cs:
            head = _fit(X[tr], y[tr], C, sw, seed)
            scores[C].append(roc_auc_score(y[va], head.logit(X[va])))
    mean_scores = {C: float(np.mean(s)) if s else float("nan") for C, s in scores.items()}
    best = max(mean_scores, key=lambda c: -np.inf if np.isnan(mean_scores[c]) else mean_scores[c])
    return _fit(X, y, best, sample_weight, seed), {"C": best, "cv_auc": mean_scores, "cv_folds": n_splits}
