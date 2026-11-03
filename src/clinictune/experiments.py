"""Leave-one-site-out (LOSO) adaptation experiments and shortcut tests.

LOSO protocol for each held-out site s:
  * train + calibrate on all other sites (patient-level splits inside :func:`train_task`);
  * split site s by patient into an *adaptation pool* (50%) and a *test* half (50%) - fixed per repeat;
  * k = 0: source calibration evaluated on the test half (the "before" row - negative results expected);
  * k in {10, 50, 100}: sample k labelled images from the pool, recalibrate (MAP Platt + threshold +
    class-conditional conformal), evaluate on the test half;
  * k >= 40 with >= 10 per class: a local residual head on top of the source head (ClinicTune's ``adapt``
    strategy), calibrated on out-of-fold logits - the only strategy that can improve ranking;
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
from clinictune.model.adapt import select_and_fit
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


def _age_band(age: str) -> str:
    if not age:
        return "unknown"
    a = int(age)
    return "<35" if a < 35 else ("35-54" if a < 55 else "55+")


def _subgroup_rows(
    records: Sequence[Record],
    idx: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    thr: float,
    sets: np.ndarray,
    meta: dict[str, Any],
) -> list[dict]:
    """Per-subgroup sensitivity / specificity / class-conditional coverage on a test set."""
    groups = {
        "sex": np.array([records[i].sex or "unknown" for i in idx]),
        "age": np.array([_age_band(records[i].age) for i in idx]),
    }
    covered = sets[np.arange(len(y)), y]
    out = []
    for var, g in groups.items():
        for level in sorted(set(g.tolist())):
            m = g == level
            if level == "unknown" or m.sum() < 10:
                continue
            pos, neg = m & (y == 1), m & (y == 0)
            out.append(
                {
                    **meta,
                    "variable": var,
                    "group": level,
                    "n": int(m.sum()),
                    "n_pos": int(pos.sum()),
                    "sensitivity": float((p[pos] >= thr).mean()) if pos.any() else None,
                    "specificity": float((p[neg] < thr).mean()) if neg.any() else None,
                    "coverage_pos": float(covered[pos].mean()) if pos.any() else None,
                    "coverage_neg": float(covered[neg].mean()) if neg.any() else None,
                }
            )
    return out


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
    adapt: bool = True,
    subgroup_rows: list[dict] | None = None,
) -> list[dict]:
    """See module docstring. ``site_adapt`` rows (local residual head, out-of-fold calibration) are added
    when the k-label sample has >= 10 images of each class; subgroup rows for the largest k are appended to
    ``subgroup_rows`` when given."""
    y, patients, sites = records_arrays(records)
    rows: list[dict] = []
    for s in sorted(set(sites.tolist())):
        src = np.flatnonzero(sites != s)
        if len(np.unique(y[src])) < 2:
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
            feasible = [k for k in ks if k <= len(pool)]
            for k in feasible:
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
                if subgroup_rows is not None and k == feasible[-1]:
                    subgroup_rows += _subgroup_rows(
                        records,
                        test,
                        y[test],
                        p,
                        thr,
                        conf.predict_sets(p),
                        {"task": task, "held_out_site": s, "k_labels": k, "repeat": rep},
                    )
                if adapt and np.bincount(y[sel], minlength=2).min() >= 10:
                    fit = select_and_fit(tm.head, X[sel], y[sel], patients[sel], seed)
                    if fit is not None:
                        residual, oof, _ = fit
                        a_platt, a_thr, a_conf = _fit_site(tm, oof, y[sel], alpha, 0.0)
                        pa = a_platt(logit_test + residual.logit(tm.head, X[test]))
                        rows.append(
                            _row(
                                s,
                                task,
                                k,
                                "site_adapt",
                                rep,
                                binary_report(y[test], pa, a_thr, n_boot=0),
                                a_conf.coverage(pa, y[test]),
                            )
                            | {"n_images": len(sel), "lambda": residual.lam}
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


def summarize_subgroups(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["task"], r["held_out_site"], r["variable"], r["group"]), []).append(r)
    out = []
    for (task, site, var, level), rs in sorted(groups.items()):
        row: dict[str, Any] = {
            "task": task,
            "held_out_site": site,
            "variable": var,
            "group": level,
            "k_labels": rs[0]["k_labels"],
            "repeats": len(rs),
            "n_per_repeat": rs[0]["n"],
        }
        for key in ("sensitivity", "specificity", "coverage_pos", "coverage_neg"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                row[key] = float(np.mean(vals))
                row[key + "_sd"] = float(np.std(vals))
        out.append(row)
    return out


def lungs_removed(mask_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Transform that fills the (manually segmented) lung fields with the mean of the rest of the image."""
    from PIL import Image

    def fn(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        left = np.asarray(Image.open(mask_path).convert("L").resize((w, h)))
        right_path = mask_path.parent.parent / "rightMask" / mask_path.name
        right = np.asarray(Image.open(right_path).convert("L").resize((w, h)))
        lung = (left > 127) | (right > 127)
        out = rgb.copy()
        out[lung] = rgb[~lung].mean(axis=0).astype(rgb.dtype)
        return out

    return fn


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


def _compose(first: Callable[[np.ndarray], np.ndarray], then: Callable[[np.ndarray], np.ndarray]):  # type: ignore[no-untyped-def]
    def fn(rgb: np.ndarray) -> np.ndarray:
        return then(first(rgb))

    return fn


def background_only(rgb: np.ndarray) -> np.ndarray:
    """Fundus version of the border test: blank the whole circular retinal field and keep only the background,
    where camera overlays, text and site-specific vignetting live. (A plain border strip is not enough for
    fundus photos - the retina often reaches the image edge.)"""
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_fill_holes

    # The backbone sees <= 518 px; masking a 12-megapixel original first would only cost time.
    img = Image.fromarray(rgb)
    img.thumbnail((512, 512), Image.Resampling.BILINEAR)
    rgb = np.asarray(img)
    gray = rgb.astype(np.float32).mean(axis=2)
    fov = binary_fill_holes(gray > 20)
    fov = binary_dilation(fov, iterations=max(2, min(rgb.shape[:2]) // 100))
    out = rgb.copy()
    out[fov] = np.median(rgb[fov], axis=0).astype(rgb.dtype) if fov.any() else out[fov]
    return out


def shortcut_test(
    records: Sequence[Record],
    tm: TaskModel,
    backbone: OnnxBackbone,
    cache_dir: str | Path,
    lung_masks: dict[str, Path] | None = None,
    max_per_site: int | None = 300,
    original_cache: str | Path | None = None,
    seed: int = 0,
    view: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict]:
    """ROC-AUC of the trained head under image perturbations, per site.

    ``max_per_site`` samples (stratified by label) to bound CPU time - every perturbation needs a fresh
    backbone pass. ``original_cache`` lets the unperturbed condition reuse embeddings already extracted.

    border_only AUC well above 0.5 => the model can predict from the image border (markers, text, collimation).
    A large drop under center_crop points the same way. ``lung_masks`` (image path -> left-lung mask, e.g.
    Montgomery's manual masks) adds ``lungs_removed``: if the model still separates TB from normal with both
    lung fields blanked out, it is reading something other than the lungs.
    """
    if max_per_site:
        rng = np.random.default_rng(seed)
        keep: list[int] = []
        for s in sorted({r.site for r in records}):
            for lab in (0, 1):
                idx = [i for i, r in enumerate(records) if r.site == s and r.label == lab]
                take = int(round(max_per_site * len(idx) / max(1, sum(r.site == s for r in records))))
                keep += sorted(rng.choice(idx, min(len(idx), max(take, 1)), replace=False).tolist()) if idx else []
        masked = {i for i, r in enumerate(records) if lung_masks and r.image_path in lung_masks}
        records = [records[i] for i in sorted(set(keep) | masked)]
    y, _, sites = records_arrays(records)
    conds: dict[str, Callable[[np.ndarray], np.ndarray] | None] = {
        "original": None,
        ("background_only" if tm.task == "dr" else "border_only"): background_only if tm.task == "dr" else border_only,
        "center_crop_80": center_crop,
    }
    rows = []
    name = backbone.spec.name
    for cond, fn in conds.items():
        # Key the cache on the transform's bytecode so editing a transform never reuses stale embeddings.
        tag = hashlib.sha256(fn.__code__.co_code).hexdigest()[:8] if fn else "none"
        is_original = fn is None
        if view is not None:  # perturb the original, then apply the model view the deployed model uses
            tag += "_" + view.__name__
            fn = _compose(fn, view) if fn else view
        path = (
            Path(original_cache)
            if (is_original and original_cache)
            else (Path(cache_dir) / f"shortcut_{name.replace('@', '-')}_{tm.task}_{cond}_{tag}.npz")
        )
        X = extract(records, backbone, path, transform=fn)
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
    if lung_masks:
        from clinictune import imaging

        idx = [i for i, r in enumerate(records) if r.image_path in lung_masks]
        embs = [
            backbone.embed([lungs_removed(lung_masks[records[i].image_path])(imaging.load(records[i].image_path))])[0][
                0
            ]
            for i in idx
        ]
        z = tm.head.logit(np.stack(embs))
        orig_path = Path(original_cache) if original_cache else Path(cache_dir) / f"shortcut_{tm.task}_masked_orig.npz"
        z_orig = tm.head.logit(extract([records[i] for i in idx], backbone, orig_path))
        ys, ss = y[idx], sites[idx]
        for s in sorted(set(ss.tolist())):
            m = ss == s
            if len(np.unique(ys[m])) == 2:
                rows.append(
                    {
                        "task": tm.task,
                        "site": s,
                        "condition": "original_masked_subset",
                        "roc_auc": float(roc_auc_score(ys[m], z_orig[m])),
                        "n": int(m.sum()),
                    }
                )
                rows.append(
                    {
                        "task": tm.task,
                        "site": s,
                        "condition": "lungs_removed",
                        "roc_auc": float(roc_auc_score(ys[m], z[m])),
                        "n": int(m.sum()),
                    }
                )
    return rows
