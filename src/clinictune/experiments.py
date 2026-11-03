"""Leave-one-site-out (LOSO) adaptation experiments and shortcut tests.

LOSO protocol for each held-out site s:
  * train + calibrate on all other sites (patient-level splits inside :func:`train_task`);
  * split site s by patient into an *adaptation pool* (50%) and a *test* half (50%) - fixed per repeat;
  * k = 0: source calibration evaluated on the test half (the "before" row - negative results expected);
  * k in {10, 50, 100}: sample k labelled images from the pool, recalibrate (MAP Platt + threshold +
    class-conditional conformal), evaluate on the test half;
  * also: a linear head trained only on the k site labels (threshold-free ROC/PR-AUC), the classic
    "few-shot linear probe" baseline.
The test half is never used for fitting or model selection.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from clinictune.data.manifest import Record
from clinictune.data.splits import patient_split, records_arrays, stratified_label_sample
from clinictune.embed.backbone import OnnxBackbone
from clinictune.embed.cache import extract
from clinictune.model.bundle import TaskModel
from clinictune.model.heads import fit_head
from clinictune.model.metrics import binary_report
from clinictune.model.train import train_task
from clinictune.site.onboarding import _fit_site

METRIC_KEYS = ("sensitivity", "specificity", "roc_auc", "pr_auc", "ece")


def _row(site: str, task: str, k: int, strategy: str, rep: int, rep_m: dict, cov: dict | None) -> dict:
    row: dict[str, Any] = {"task": task, "held_out_site": site, "k_labels": k, "strategy": strategy, "repeat": rep}
    for key in METRIC_KEYS:
        row[key] = rep_m.get(key)
    if cov:
        row["coverage_neg"] = cov.get("class_0")
        row["coverage_pos"] = cov.get("class_1")
        row["singleton_rate"] = cov.get("singleton_rate")
    return row


def loso(
    X: np.ndarray,
    records: Sequence[Record],
    task: str,
    *,
    ks: Sequence[int] = (10, 50, 100),
    repeats: int = 10,
    alpha: float = 0.1,
    prior_strength: float = 5.0,
    seed: int = 0,
) -> list[dict]:
    y, patients, sites = records_arrays(records)
    rows: list[dict] = []
    for s in sorted(set(sites.tolist())):
        src = np.flatnonzero(sites != s)
        if len(np.unique(sites[src])) < 1 or len(np.unique(y[src])) < 2:
            continue
        tm: TaskModel = train_task(X[src], [records[i] for i in src], task, alpha=alpha, seed=seed)
        tgt = np.flatnonzero(sites == s)
        for rep in range(repeats):
            pool_i, test_i = patient_split(y[tgt], patients[tgt], [0.5, 0.5], seed=seed + rep)
            pool, test = tgt[pool_i], tgt[test_i]
            logit_test = tm.head.logit(X[test])
            p0 = tm.platt(logit_test)
            rows.append(
                _row(
                    s,
                    task,
                    0,
                    "source_calibration",
                    rep,
                    binary_report(y[test], p0, tm.threshold, n_boot=0),
                    tm.conformal.coverage(p0, y[test]),
                )
            )
            rng = np.random.default_rng(seed * 1000 + rep)
            for k in ks:
                if k > len(pool):
                    continue
                sel = pool[stratified_label_sample(y[pool], patients[pool], k, rng)]
                if len(np.unique(y[sel])) < 2:
                    continue
                platt, thr, conf = _fit_site(tm, tm.head.logit(X[sel]), y[sel], alpha, prior_strength)
                p = platt(logit_test)
                rows.append(
                    _row(
                        s,
                        task,
                        k,
                        "site_recalibration",
                        rep,
                        binary_report(y[test], p, thr, n_boot=0),
                        conf.coverage(p, y[test]),
                    )
                    | {"n_images": len(sel)}
                )
                head, _ = fit_head(X[sel], y[sel], patients[sel], seed=seed)
                z = head.logit(X[test])
                rows.append(
                    {
                        "task": task,
                        "held_out_site": s,
                        "k_labels": k,
                        "n_images": len(sel),
                        "strategy": "site_only_head",
                        "repeat": rep,
                        "roc_auc": float(roc_auc_score(y[test], z)),
                        "pr_auc": float(average_precision_score(y[test], z)),
                    }
                )
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    """Mean and SD over repeats, grouped by (task, site, strategy, k)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["task"], r["held_out_site"], r["strategy"], r["k_labels"]), []).append(r)
    out = []
    for (task, site, strategy, k), rs in sorted(groups.items()):
        row: dict[str, Any] = {
            "task": task,
            "held_out_site": site,
            "strategy": strategy,
            "k_labels": k,
            "repeats": len(rs),
        }
        for key in (*METRIC_KEYS, "coverage_neg", "coverage_pos", "singleton_rate"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                row[key] = float(np.mean(vals))
                row[key + "_sd"] = float(np.std(vals))
        out.append(row)
    return out


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def markdown_table(summary: list[dict], alpha: float = 0.1) -> str:
    def f(r: dict, k: str) -> str:
        if k not in r:
            return "-"
        sd = r.get(k + "_sd")
        return f"{r[k]:.2f} ± {sd:.2f}" if sd is not None else f"{r[k]:.2f}"

    lines = [
        f"| task | held-out site | strategy | k | sens | spec | ROC-AUC | PR-AUC | ECE | "
        f"cov(neg) | cov(pos) | target cov {1 - alpha:.2f} |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        miss = [c for c in ("coverage_neg", "coverage_pos") if c in r and r[c] < (1 - alpha) - 0.05]
        flag = "**FAILS** " + ",".join(m.split("_")[1] for m in miss) if miss else ("ok" if "coverage_neg" in r else "")
        lines.append(
            f"| {r['task']} | {r['held_out_site']} | {r['strategy']} | {r['k_labels']} | {f(r, 'sensitivity')} | "
            f"{f(r, 'specificity')} | {f(r, 'roc_auc')} | {f(r, 'pr_auc')} | {f(r, 'ece')} | "
            f"{f(r, 'coverage_neg')} | {f(r, 'coverage_pos')} | {flag} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- shortcut tests


def border_only(rgb: np.ndarray, keep: float = 0.12) -> np.ndarray:
    """Grey out everything except a border strip: if the model still discriminates, it uses non-anatomical cues."""
    out = rgb.copy()
    h, w = rgb.shape[:2]
    bh, bw = int(h * keep), int(w * keep)
    # Fill with the border's own mean: the whole-image mean would leak lesion brightness into the fill.
    border = np.ones((h, w), dtype=bool)
    border[bh : h - bh, bw : w - bw] = False
    out[~border] = rgb[border].mean(axis=0).astype(rgb.dtype)
    return out


def center_crop(rgb: np.ndarray, frac: float = 0.8) -> np.ndarray:
    h, w = rgb.shape[:2]
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return rgb[dh : h - dh, dw : w - dw]


def shortcut_test(
    records: Sequence[Record],
    tm: TaskModel,
    backbone: OnnxBackbone,
    cache_dir: str | Path,
    mask_fn: Callable[[Record], Callable[[np.ndarray], np.ndarray]] | None = None,
) -> list[dict]:
    """ROC-AUC of the trained head under image perturbations, per site.

    border_only AUC well above 0.5 => the model can predict from the image border (markers, text, collimation).
    A large drop under center_crop points the same way. ``mask_fn`` (e.g. lung masks for Montgomery) adds a
    lungs-removed condition.
    """
    y, _, sites = records_arrays(records)
    conds: dict[str, Callable[[np.ndarray], np.ndarray] | None] = {
        "original": None,
        "border_only": border_only,
        "center_crop_80": center_crop,
    }
    rows = []
    name = backbone.spec.name
    for cond, fn in conds.items():
        # Key the cache on the transform's bytecode so editing a transform never reuses stale embeddings.
        tag = hashlib.sha256(fn.__code__.co_code).hexdigest()[:8] if fn else "none"
        X = extract(records, backbone, Path(cache_dir) / f"shortcut_{name}_{tm.task}_{cond}_{tag}.npz", transform=fn)
        z = tm.head.logit(X)
        for s in sorted(set(sites.tolist())):
            m = sites == s
            if len(np.unique(y[m])) == 2:
                rows.append(
                    {
                        "task": tm.task,
                        "site": s,
                        "condition": cond,
                        "roc_auc": float(roc_auc_score(y[m], z[m])),
                        "n": int(m.sum()),
                    }
                )
    if mask_fn is not None:
        # Per-record transforms can't share one extract() call's transform; extract one by one.
        from clinictune import imaging

        embs = []
        for r in records:
            rgb = mask_fn(r)(imaging.load(r.image_path))
            embs.append(backbone.embed([rgb])[0][0])
        z = tm.head.logit(np.stack(embs))
        for s in sorted(set(sites.tolist())):
            m = sites == s
            if len(np.unique(y[m])) == 2:
                rows.append(
                    {
                        "task": tm.task,
                        "site": s,
                        "condition": "lungs_masked",
                        "roc_auc": float(roc_auc_score(y[m], z[m])),
                        "n": int(m.sum()),
                    }
                )
    return rows
