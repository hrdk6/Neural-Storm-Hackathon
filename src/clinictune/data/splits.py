"""Patient-level splitting. Every split in the project goes through here so leakage is checked once."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from clinictune.data.manifest import Record


class LeakageError(RuntimeError):
    pass


def assert_no_patient_leakage(*groups: Sequence[str] | np.ndarray, names: Sequence[str] | None = None) -> None:
    sets = [set(map(str, g)) for g in groups]
    names = list(names or [f"split{i}" for i in range(len(sets))])
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if shared:
                example = sorted(shared)[:3]
                raise LeakageError(f"{len(shared)} patients in both {names[i]} and {names[j]}, e.g. {example}")


def patient_split(
    labels: np.ndarray,
    patients: np.ndarray,
    fractions: Sequence[float],
    seed: int,
) -> list[np.ndarray]:
    """Split indices into len(fractions) parts, grouped by patient and stratified by label.

    Uses StratifiedGroupKFold with enough folds to approximate the requested fractions, then assigns
    whole folds to parts. Returns index arrays; asserts no patient appears in two parts.
    """
    fractions = np.asarray(fractions, dtype=float)
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("fractions must sum to 1")
    n_folds = 20 if len(np.unique(patients)) >= 40 else max(2, len(np.unique(patients)) // 2)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [test for _, test in sgkf.split(np.zeros(len(labels)), labels, patients)]
    edges = [0]
    for i, f in enumerate(np.cumsum(fractions)):
        e = int(round(f * n_folds))
        if fractions[i] > 0:
            e = max(e, edges[-1] + 1)
        edges.append(min(e, n_folds))
    edges[-1] = n_folds
    parts = [
        np.sort(np.concatenate(folds[a:b])) if b > a else np.array([], dtype=int)
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert_no_patient_leakage(*[patients[p] for p in parts])
    return parts


def stratified_label_sample(labels: np.ndarray, patients: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ~n images (whole patients) keeping the label mix; guarantees >=1 per class when possible.

    Used to simulate "a clinic labels n local images" in the leave-one-site-out experiments.
    """
    uniq = np.unique(patients)
    patient_label = {p: labels[patients == p].max() for p in uniq}
    by_class = {c: [p for p in uniq if patient_label[p] == c] for c in (0, 1)}
    prevalence = np.mean([patient_label[p] for p in uniq])
    n = max(2, round(n * len(uniq) / len(patients)))  # images -> patients
    n_pos = int(np.clip(round(n * prevalence), 1 if by_class[1] else 0, len(by_class[1])))
    n_neg = int(np.clip(n - n_pos, 1 if by_class[0] else 0, len(by_class[0])))
    chosen = list(rng.choice(by_class[1], n_pos, replace=False)) if n_pos else []
    chosen += list(rng.choice(by_class[0], n_neg, replace=False)) if n_neg else []
    return np.flatnonzero(np.isin(patients, chosen))


def records_arrays(records: Sequence[Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([r.label for r in records], dtype=int)
    patients = np.array([r.patient_id for r in records])
    sites = np.array([r.site for r in records])
    return labels, patients, sites
