"""End-to-end builders: train a bundle from a manifest, or build the synthetic demo bundle."""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from clinictune import imaging, synthetic
from clinictune.data.manifest import Record, read_manifest, summarize, write_manifest
from clinictune.data.splits import patient_split
from clinictune.embed.backbone import BackboneSpec, OnnxBackbone, sha256_file
from clinictune.embed.cache import extract
from clinictune.embed.demo_backbone import build_demo_backbone
from clinictune.model.bundle import CaseLibrary, GradabilityHead, backbone_dir, save_bundle
from clinictune.model.calibration import fit_platt
from clinictune.model.heads import fit_head
from clinictune.model.metrics import binary_report
from clinictune.model.train import train_task

log = logging.getLogger(__name__)


def load_backbone_dir(path: str | Path, threads: int = 0) -> OnnxBackbone:
    path = Path(path)
    spec = BackboneSpec.from_dict(json.loads((path / "spec.json").read_text(encoding="utf-8")))
    return OnnxBackbone(path, spec, threads=threads)


def write_backbone_spec(path: str | Path, spec: BackboneSpec) -> None:
    Path(path, "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- gradability + case library


def train_gradability(
    items: Sequence[tuple[str, int]],
    backbone: OnnxBackbone,
    cache: Path,
    max_reject_gradable: float = 0.05,
    seed: int = 0,
    transform=None,  # type: ignore[no-untyped-def]
) -> GradabilityHead:
    """Linear 'ungradable' head on the task backbone's embeddings.

    The threshold rejects at most ``max_reject_gradable`` of gradable images on held-out data (a gate that
    bounces good images costs a clinic real time); sensitivity for ungradable images is reported honestly.
    """
    recs = [Record(p, y, f"q:{Path(p).stem}", "quality", "dr", "quality") for p, y in items]
    X = extract(recs, backbone, cache, transform=transform)
    y = np.array([y for _, y in items])
    groups = np.array([r.patient_id for r in recs])
    tr, cal, te = patient_split(y, groups, [0.6, 0.2, 0.2], seed=seed)
    head, info = fit_head(X[tr], y[tr], groups[tr], seed=seed)
    platt = fit_platt(head.logit(X[cal]), y[cal])
    p_cal = platt(head.logit(X[cal]))
    gradable_scores = np.sort(p_cal[y[cal] == 0])
    threshold = float(gradable_scores[int(np.ceil((1 - max_reject_gradable) * len(gradable_scores))) - 1])
    p_te = platt(head.logit(X[te]))
    rep = binary_report(y[te], p_te, threshold, n_boot=200)
    return GradabilityHead(
        head,
        platt,
        threshold,
        {
            "test": rep,
            "note": "positive class = ungradable; threshold rejects <= 5% of gradable images on calibration data.",
            "n": {"train": len(tr), "calibration": len(cal), "test": len(te)},
            "head_selection": info,
        },
    )


def build_library(
    records: Sequence[Record], X: np.ndarray, out_root: Path, task: str, size: int = 240, seed: int = 0
) -> CaseLibrary:
    """Stratified (site x label) sample of training cases with 192px thumbnails written into the bundle."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, int], list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault((r.site, r.label), []).append(i)
    per = max(1, size // max(len(groups), 1))
    chosen = sorted(int(j) for idx in groups.values() for j in rng.choice(idx, min(per, len(idx)), replace=False))
    thumbs = out_root / "library" / task
    thumbs.mkdir(parents=True, exist_ok=True)
    meta: list[dict[str, Any]] = []
    for k, i in enumerate(chosen):
        r = records[i]
        img = Image.fromarray(imaging.load(r.image_path))
        img.thumbnail((192, 192), Image.Resampling.LANCZOS)
        name = f"{k:04d}.jpg"
        img.convert("RGB").save(thumbs / name, quality=82, optimize=True)
        meta.append(
            {"dataset": r.dataset, "site": r.site, "thumb": f"library/{task}/{name}", "sex": r.sex, "age": r.age}
        )
    emb = X[chosen]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return CaseLibrary(emb.astype(np.float32), np.array([records[i].label for i in chosen]), meta)


# ----------------------------------------------------------------------------- training


def train_bundle(
    manifest: str | Path,
    backbone_dirs: dict[str, str | Path],
    bundle_dir: str | Path,
    *,
    tasks: list[str],
    exclude_sites: list[str] | None = None,
    cache_dir: str | Path = "cache",
    alpha: float = 0.1,
    seed: int = 0,
    demo: bool = False,
    gradability: dict[str, Sequence[tuple[str, int]]] | None = None,
    library_size: int = 240,
    notes: str = "",
) -> dict:
    """Train heads for ``tasks`` on every site in the manifest except ``exclude_sites`` (held out for
    onboarding / external testing - never tuned on). ``backbone_dirs`` maps task -> exported backbone dir."""
    records = read_manifest(manifest)
    exclude = set(exclude_sites or [])
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, BackboneSpec] = {}
    task_models = {}
    for task in tasks:
        backbone = load_backbone_dir(backbone_dirs[task])
        name = backbone.spec.name
        if name not in specs:
            dest = backbone_dir(bundle_dir, name)
            dest.mkdir(parents=True, exist_ok=True)
            # Skip identical weights: re-training a bundle must not rewrite (possibly memory-mapped) files.
            if not ((dest / "backbone.onnx").exists() and sha256_file(dest / "backbone.onnx") == backbone.spec.sha256):
                for f in Path(backbone_dirs[task]).glob("backbone.onnx*"):
                    shutil.copy2(f, dest / f.name)
            specs[name] = backbone.spec
        recs = [r for r in records if r.task == task and r.site not in exclude]
        if not recs:
            raise ValueError(f"no training records for task {task}")
        log.info("task %s (%s): %s", task, name, summarize(recs))
        pre = imaging.PREPROCESS.get(task, "") if not demo else ""
        transform = imaging.view_transform(pre)
        tag = f"_{imaging.view_tag(pre)}" if pre else ""
        cache = Path(cache_dir) / f"{_safe(name)}_{task}{tag}_all.npz"  # shared with `loso`
        X = extract(recs, backbone, cache, transform=transform)
        tm = train_task(X, recs, task, alpha=alpha, seed=seed, backbone=name)
        tm.preprocess = pre
        if library_size:
            tm.library = build_library(recs, X, bundle_dir, task, size=library_size, seed=seed)
        if gradability and task in gradability:
            log.info("task %s: training gradability gate on %d images", task, len(gradability[task]))
            tm.gradability = train_gradability(gradability[task], backbone, cache, seed=seed, transform=transform)
        task_models[task] = tm
    datasets = sorted({r.dataset for r in records if r.site not in exclude and r.task in tasks})
    return save_bundle(
        bundle_dir,
        specs,
        task_models,
        datasets=datasets,
        demo=demo,
        notes=notes or f"held-out sites: {sorted(exclude)}",
    )


def _safe(name: str) -> str:
    return name.replace("@", "-").replace("/", "-")


# ----------------------------------------------------------------------------- synthetic demo


def build_demo(root: str | Path = ".", n_scale: float = 1.0, seed: int = 0) -> dict[str, Path]:
    """Synthetic data -> demo backbone -> demo bundle -> onboarding packs for two 'new clinics'."""
    root = Path(root)
    data_dir = root / "data" / "synthetic"
    bb_dir = root / "models" / "backbones" / "demo-randproj"
    bundle_dir = root / "models" / "demo"

    records: list[Record] = []
    for i, p in enumerate(synthetic.TRAIN_SITES + synthetic.NEW_CLINICS):
        floor = 60 if p in synthetic.NEW_CLINICS else 40
        p = synthetic.SiteProfile(**{**p.__dict__, "n": max(floor, int(p.n * n_scale))})
        log.info("generating %s (%d images)", p.name, p.n)
        records += synthetic.generate_site(p, data_dir, seed=seed + i)
    manifest = data_dir / "manifest.csv"
    write_manifest(records, manifest)

    spec = build_demo_backbone(bb_dir, seed=seed)
    write_backbone_spec(bb_dir, spec)
    new_sites = [p.name for p in synthetic.NEW_CLINICS]
    train_bundle(
        manifest,
        {"tb": bb_dir, "dr": bb_dir},
        bundle_dir,
        tasks=["tb", "dr"],
        exclude_sites=new_sites,
        cache_dir=root / "cache",
        seed=seed,
        demo=True,
        library_size=48,
    )
    packs = {}
    for site in new_sites:
        recs = [r for r in records if r.site == site]
        n_onboard = min(100, len(recs) - max(10, len(recs) // 3))
        packs[site] = make_onboarding_zip(recs[:n_onboard], root / "data" / "onboarding" / f"{site}.zip")
        sample_dir = root / "data" / "onboarding" / f"{site}_unseen"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for r in recs[n_onboard:]:
            shutil.copy2(r.image_path, sample_dir / f"label{r.label}_{Path(r.image_path).name}")
    return {"manifest": manifest, "bundle": bundle_dir, **packs}


def make_onboarding_zip(records: Sequence[Record], out: str | Path, max_side: int | None = None) -> Path:
    """Onboarding pack format: images + labels.csv (filename,label[,patient_id,sex,age]).

    ``max_side`` downsizes images (to high-quality JPEG) to keep packs uploadable from a clinic connection.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        rows = [["filename", "label", "patient_id", "sex", "age"]]
        for r in records:
            name = Path(r.image_path).name
            if max_side:
                img = Image.fromarray(imaging.load(r.image_path))
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                name = Path(name).with_suffix(".jpg").name
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
                zf.writestr(f"images/{name}", buf.getvalue())
            else:
                zf.write(r.image_path, f"images/{name}")
            rows.append([name, str(r.label), r.patient_id.split(":")[-1], r.sex, r.age])
        tmp = out.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        zf.write(tmp, "labels.csv")
        tmp.unlink()
    return out
