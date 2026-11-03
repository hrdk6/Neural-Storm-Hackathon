"""Dataset-shift measurement between the training sites and a new clinic.

Three complementary views:
* covariate shift in embedding space - a domain classifier's AUC (0.5 = indistinguishable) and an RBF
  MMD two-sample test with a permutation p-value;
* label shift - observed local prevalence vs. training prevalence, plus a black-box (BBSE) estimate that
  needs no labels and is used for ongoing monitoring;
* performance shift - computed by the onboarding flow from local labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def domain_classifier_auc(source: np.ndarray, target: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    s = _subsample(source, max(3 * len(target), 200), rng)
    X = np.vstack([s, target])
    y = np.r_[np.zeros(len(s)), np.ones(len(target))]
    k = int(min(32, X.shape[1], len(X) - 1))
    Z = PCA(n_components=k, random_state=seed).fit_transform(X)
    n_splits = int(min(5, len(target)))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    proba = cross_val_predict(clf, Z, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    d = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd_test(source: np.ndarray, target: np.ndarray, n_perm: int = 200, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    s = _subsample(source, 400, rng)
    t = _subsample(target, 400, rng)
    Z = np.vstack([s, t]).astype(np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    sq = (Z**2).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    med = np.median(d[np.triu_indices(len(Z), 1)])
    K = _rbf(Z, Z, 1.0 / max(med, 1e-8))
    n = len(s)

    def stat(idx: np.ndarray) -> float:
        a, b = idx[:n], idx[n:]
        kxx = K[np.ix_(a, a)]
        kyy = K[np.ix_(b, b)]
        kxy = K[np.ix_(a, b)]
        m, k = len(a), len(b)
        return float(
            (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (k * (k - 1)) - 2 * kxy.mean()
        )

    base = np.arange(len(Z))
    observed = stat(base)
    perms = np.array([stat(rng.permutation(base)) for _ in range(n_perm)])
    return {"mmd2": observed, "p_value": float((1 + np.sum(perms >= observed)) / (1 + n_perm))}


def bbse_prevalence(p_target: np.ndarray, threshold: float, sens: float, spec: float) -> float | None:
    """Black-box shift estimation (Lipton et al. 2018) for binary labels:
    P(pred=1) = sens * pi + (1 - spec) * (1 - pi)  =>  pi = (q - (1 - spec)) / (sens + spec - 1)."""
    denom = sens + spec - 1
    if denom <= 0.05:
        return None
    q = float(np.mean(np.asarray(p_target) >= threshold))
    return float(np.clip((q - (1 - spec)) / denom, 0.0, 1.0))


def severity(domain_auc: float) -> str:
    if np.isnan(domain_auc):
        return "unknown"
    if domain_auc < 0.6:
        return "none"
    if domain_auc < 0.75:
        return "mild"
    if domain_auc < 0.9:
        return "moderate"
    return "severe"


def shift_report(source: np.ndarray, target: np.ndarray, seed: int = 0) -> dict[str, Any]:
    auc = domain_classifier_auc(source, target, seed)
    mmd = mmd_test(source, target, seed=seed)
    return {"domain_auc": auc, "severity": severity(auc), **mmd, "n_source": len(source), "n_site": len(target)}
