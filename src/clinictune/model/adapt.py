"""Local residual head: few-shot site adaptation that can fix *ranking*, not just calibration.

Recalibration (Platt + threshold) is monotone, so it can never improve ROC-AUC at a new site. When a site
has enough labels, ClinicTune also fits a small L2-regularised residual on top of the frozen training-site
head, in the head's own standardised feature space:

    logit_site(x) = logit_source(x) + w . (x - mu) / sigma + b

The source logit is a fixed offset, so with few labels (or a large penalty) the model stays close to the
training-site model, and with more labels it learns what is different here. The penalty is chosen by inner
cross-validation; calibration is fitted on *out-of-fold* logits so it is not fooled by in-sample fit.
Because the residual is linear in the mean-pooled tokens, the exact per-patch heatmap still holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from clinictune.model.heads import LinearHead

LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass
class ResidualHead:
    coef: np.ndarray
    intercept: float
    lam: float

    def logit(self, head: LinearHead, emb: np.ndarray) -> np.ndarray:
        return ((emb - head.mean) / head.scale) @ self.coef + self.intercept

    def token_contributions(self, head: LinearHead, tokens: np.ndarray) -> np.ndarray:
        return ((tokens - head.mean) / head.scale) @ self.coef

    def to_dict(self) -> dict[str, Any]:
        return {"coef": self.coef.tolist(), "intercept": self.intercept, "lam": self.lam}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResidualHead:
        return cls(np.asarray(d["coef"], dtype=np.float32), float(d["intercept"]), float(d["lam"]))


def fit_residual(head: LinearHead, emb: np.ndarray, y: np.ndarray, lam: float) -> ResidualHead:
    Z = ((emb - head.mean) / head.scale).astype(np.float64)
    offset = head.logit(emb).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1 = max(y.sum(), 1.0)
    n0 = max(len(y) - y.sum(), 1.0)
    s = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))  # balanced class weights

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:-1], theta[-1]
        z = offset + Z @ w + b
        loss = np.sum(s * (np.logaddexp(0, z) - y * z)) + 0.5 * lam * w @ w
        r = s * (expit(z) - y)
        return float(loss), np.r_[Z.T @ r + lam * w, r.sum()]

    res = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return ResidualHead(res.x[:-1].astype(np.float32), float(res.x[-1]), lam)


def _folds(y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = int(min(k, np.bincount(y, minlength=2).min(), len(np.unique(groups))))
    if n_splits < 2:
        return []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def select_and_fit(
    head: LinearHead, emb: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0
) -> tuple[ResidualHead, np.ndarray, dict[str, float]] | None:
    """Choose lambda by inner CV ROC-AUC; return (residual fitted on all data, out-of-fold logits for the
    chosen lambda, per-lambda CV AUC). None if there are too few labels for an inner split."""
    folds = _folds(y, groups, 4, seed)
    if not folds:
        return None
    base = head.logit(emb)
    oof = {lam: np.zeros(len(y)) for lam in LAMBDAS}
    for tr, te in folds:
        for lam in LAMBDAS:
            r = fit_residual(head, emb[tr], y[tr], lam)
            oof[lam][te] = base[te] + r.logit(head, emb[te])
    aucs = {lam: float(roc_auc_score(y, oof[lam])) for lam in LAMBDAS}
    best = max(LAMBDAS, key=lambda lam: (round(aucs[lam], 3), lam))  # ties -> stronger regularisation
    return fit_residual(head, emb, y, best), oof[best], {str(k): v for k, v in aucs.items()}
