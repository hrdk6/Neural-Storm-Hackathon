"""Train one task model from cached embeddings.

Patient-level split of the *training sites only*: 65% head / 10% Platt / 10% conformal / 15% internal test.
Platt and conformal use disjoint data so the conformal quantile is computed on a fixed, calibrated model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from clinictune.data.manifest import Record
from clinictune.data.splits import patient_split, records_arrays
from clinictune.model.bundle import TaskModel
from clinictune.model.calibration import fit_platt, threshold_for_sensitivity
from clinictune.model.conformal import ClassConditionalConformal
from clinictune.model.heads import fit_head
from clinictune.model.metrics import binary_report
from clinictune.model.ood import OODDetector

DEFAULT_TARGET_SENS = {"tb": 0.90, "dr": 0.85}


def train_task(
    X: np.ndarray,
    records: Sequence[Record],
    task: str,
    *,
    alpha: float = 0.1,
    target_sensitivity: float | None = None,
    seed: int = 0,
) -> TaskModel:
    y, patients, sites = records_arrays(records)
    tr, pl, cf, te = patient_split(y, patients, [0.65, 0.10, 0.10, 0.15], seed=seed)
    target = target_sensitivity or DEFAULT_TARGET_SENS[task]

    head, cv_info = fit_head(X[tr], y[tr], patients[tr], seed=seed)
    platt = fit_platt(head.logit(X[pl]), y[pl])
    p_cf = platt(head.logit(X[cf]))
    threshold = threshold_for_sensitivity(p_cf, y[cf], target)
    conformal = ClassConditionalConformal.fit(p_cf, y[cf], alpha)
    ood = OODDetector.fit(X[tr], X[cf])

    p_te = platt(head.logit(X[te]))
    metrics: dict[str, Any] = {
        "internal_test_note": "Held-out patients from the training sites (not an external site).",
        "overall": binary_report(y[te], p_te, threshold, patients[te]),
        "conformal_coverage": conformal.coverage(p_te, y[te]),
        "per_site": {},
    }
    for s in np.unique(sites[te]):
        m = sites[te] == s
        metrics["per_site"][str(s)] = binary_report(y[te][m], p_te[m], threshold, patients[te][m], n_boot=200)

    rng = np.random.default_rng(seed)
    ref = X[tr][rng.choice(len(tr), min(400, len(tr)), replace=False)]
    datasets = sorted({r.dataset for r in records})
    official = sorted({f"{r.dataset}:{r.official_split}" for r in records if r.official_split})
    return TaskModel(
        task=task,
        head=head,
        platt=platt,
        threshold=threshold,
        target_sensitivity=target,
        conformal=conformal,
        ood=ood,
        reference=ref,
        metrics=metrics,
        training={
            "sites": sorted(set(sites.tolist())),
            "datasets": datasets,
            "official_splits_used": official,
            "n": {"head": len(tr), "platt": len(pl), "conformal": len(cf), "internal_test": len(te)},
            "head_selection": cv_info,
            "alpha": alpha,
            "seed": seed,
        },
    )
