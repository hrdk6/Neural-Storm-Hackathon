"""End-to-end builders: train a bundle from a manifest, or build the synthetic demo bundle."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import zipfile
from pathlib import Path

from clinictune import synthetic
from clinictune.data.manifest import Record, read_manifest, summarize, write_manifest
from clinictune.embed.backbone import BackboneSpec, OnnxBackbone
from clinictune.embed.cache import extract
from clinictune.embed.demo_backbone import build_demo_backbone
from clinictune.model.bundle import save_bundle
from clinictune.model.train import train_task

log = logging.getLogger(__name__)


def load_backbone_dir(backbone_dir: str | Path, threads: int = 0) -> OnnxBackbone:
    backbone_dir = Path(backbone_dir)
    spec = BackboneSpec.from_dict(json.loads((backbone_dir / "spec.json").read_text(encoding="utf-8")))
    return OnnxBackbone(backbone_dir, spec, threads=threads)


def write_backbone_spec(backbone_dir: str | Path, spec: BackboneSpec) -> None:
    Path(backbone_dir, "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")


def train_bundle(
    manifest: str | Path,
    backbone_dir: str | Path,
    bundle_dir: str | Path,
    *,
    tasks: list[str],
    exclude_sites: list[str] | None = None,
    cache_dir: str | Path = "cache",
    alpha: float = 0.1,
    seed: int = 0,
    demo: bool = False,
) -> dict:
    """Train heads for ``tasks`` on every site in the manifest except ``exclude_sites`` (held out for
    onboarding / external testing - never tuned on)."""
    records = read_manifest(manifest)
    exclude = set(exclude_sites or [])
    backbone = load_backbone_dir(backbone_dir)
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(backbone_dir).glob("backbone.onnx*"):
        shutil.copy2(f, bundle_dir / f.name)

    task_models = {}
    for task in tasks:
        recs = [r for r in records if r.task == task and r.site not in exclude]
        if not recs:
            raise ValueError(f"no training records for task {task}")
        log.info("task %s: %s", task, summarize(recs))
        X = extract(recs, backbone, Path(cache_dir) / f"{backbone.spec.name}_{task}.npz")
        task_models[task] = train_task(X, recs, task, alpha=alpha, seed=seed)
    datasets = sorted({r.dataset for r in records if r.site not in exclude})
    return save_bundle(
        bundle_dir, backbone.spec, task_models, datasets=datasets, demo=demo, notes=f"held-out sites: {sorted(exclude)}"
    )


def build_demo(root: str | Path = ".", n_scale: float = 1.0, seed: int = 0) -> dict[str, Path]:
    """Synthetic data -> demo backbone -> demo bundle -> onboarding packs for two 'new clinics'."""
    root = Path(root)
    data_dir = root / "data" / "synthetic"
    backbone_dir = root / "models" / "backbones" / "demo-randproj"
    bundle_dir = root / "models" / "demo"

    records: list[Record] = []
    for i, p in enumerate(synthetic.TRAIN_SITES + synthetic.NEW_CLINICS):
        floor = 60 if p in synthetic.NEW_CLINICS else 40
        p = synthetic.SiteProfile(**{**p.__dict__, "n": max(floor, int(p.n * n_scale))})
        log.info("generating %s (%d images)", p.name, p.n)
        records += synthetic.generate_site(p, data_dir, seed=seed + i)
    manifest = data_dir / "manifest.csv"
    write_manifest(records, manifest)

    spec = build_demo_backbone(backbone_dir, seed=seed)
    write_backbone_spec(backbone_dir, spec)
    new_sites = [p.name for p in synthetic.NEW_CLINICS]
    train_bundle(
        manifest,
        backbone_dir,
        bundle_dir,
        tasks=["tb", "dr"],
        exclude_sites=new_sites,
        cache_dir=root / "cache",
        seed=seed,
        demo=True,
    )

    packs = {}
    for site in new_sites:
        recs = [r for r in records if r.site == site]
        n_onboard = min(100, len(recs) - max(10, len(recs) // 3))
        packs[site] = make_onboarding_zip(recs[:n_onboard], root / "data" / "onboarding" / f"{site}.zip")
        # The rest stay unseen: use them to try the screening page.
        sample_dir = root / "data" / "onboarding" / f"{site}_unseen"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for r in recs[n_onboard:]:
            shutil.copy2(r.image_path, sample_dir / f"label{r.label}_{Path(r.image_path).name}")
    return {"manifest": manifest, "bundle": bundle_dir, **packs}


def make_onboarding_zip(records: list[Record], out: str | Path) -> Path:
    """Onboarding pack format: images + labels.csv (filename,label[,patient_id])."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        rows = [["filename", "label", "patient_id"]]
        for r in records:
            name = Path(r.image_path).name
            zf.write(r.image_path, f"images/{name}")
            rows.append([name, str(r.label), r.patient_id.split(":")[-1]])
        tmp = out.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        zf.write(tmp, "labels.csv")
        tmp.unlink()
    return out
