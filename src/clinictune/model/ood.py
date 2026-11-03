"""Out-of-distribution check in embedding space (wrong modality, wrong body part, screenshots...).

PCA on training embeddings; score = Mahalanobis distance in the top-k subspace plus the normalised
reconstruction residual. The threshold is a high percentile of held-out calibration scores, and a site
can raise it at onboarding so its own (accepted) images are not flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class OODDetector:
    mean: np.ndarray
    components: np.ndarray  # [k, D]
    var: np.ndarray  # [k]
    resid_scale: float
    threshold: float

    def score(self, emb: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(emb) - self.mean
        Z = X @ self.components.T
        maha = np.sqrt(np.sum(Z**2 / self.var, axis=1) / len(self.var))
        resid = np.linalg.norm(X - Z @ self.components, axis=1) / self.resid_scale
        return maha + resid

    @classmethod
    def fit(cls, train: np.ndarray, calib: np.ndarray, k: int = 32, percentile: float = 99.5) -> OODDetector:
        k = int(min(k, train.shape[0] - 1, train.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(train)
        X = train - pca.mean_
        Z = X @ pca.components_.T
        resid = np.linalg.norm(X - Z @ pca.components_, axis=1)
        det = cls(
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
            var=np.maximum(pca.explained_variance_, 1e-8).astype(np.float32),
            resid_scale=float(np.median(resid) + 1e-8),
            threshold=float("inf"),
        )
        det.threshold = float(np.percentile(det.score(calib), percentile))
        return det

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
            "var": self.var.tolist(),
            "resid_scale": self.resid_scale,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OODDetector:
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            components=np.asarray(d["components"], dtype=np.float32),
            var=np.asarray(d["var"], dtype=np.float32),
            resid_scale=float(d["resid_scale"]),
            threshold=float(d["threshold"]),
        )
