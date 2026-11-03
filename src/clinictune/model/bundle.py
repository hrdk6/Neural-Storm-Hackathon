"""Model bundle: one directory holding everything needed to serve.

    <bundle>/manifest.json                     bundle id, backbones, tasks, intended use, datasets
    <bundle>/backbones/<name>/backbone.onnx    frozen backbones (sha256-checked at load), one per task family
    <bundle>/tasks/<task>.json                 head, calibration, threshold, conformal, OOD, gradability, metrics
    <bundle>/tasks/<task>_ref.npy              subsample of training embeddings (shift tests at onboarding)
    <bundle>/tasks/<task>_library.npz          reference-case library for similar-case retrieval
    <bundle>/library/<task>/<i>.jpg            small thumbnails of the reference cases
    <bundle>/evidence.json                     validation tables (LOSO, shortcut, subgroups) shown in the UI

Each task names its own backbone: a chest-X-ray foundation model for TB and a general vision model for
fundus photos can live in one bundle. Bundles contain model weights and data-derived files - they are
never committed to git.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from clinictune import DISCLAIMER, __version__
from clinictune.embed.backbone import BackboneSpec, OnnxBackbone
from clinictune.model.calibration import Platt
from clinictune.model.conformal import ClassConditionalConformal
from clinictune.model.heads import LinearHead
from clinictune.model.ood import OODDetector

TASK_LABELS = {
    "tb": {"modality": "chest X-ray (PA/AP)", "finding": "findings consistent with possible tuberculosis"},
    "dr": {"modality": "colour fundus photograph", "finding": "signs of referable diabetic retinopathy"},
}


@dataclass
class GradabilityHead:
    """Learned image-quality gate: P(ungradable) from the embedding (trained on graded quality labels)."""

    head: LinearHead
    platt: Platt
    threshold: float
    metrics: dict[str, Any] = field(default_factory=dict)

    def p_ungradable(self, emb: np.ndarray) -> np.ndarray:
        return self.platt(self.head.logit(emb))

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head.to_dict(),
            "platt": self.platt.to_dict(),
            "threshold": self.threshold,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GradabilityHead:
        return cls(
            LinearHead.from_dict(d["head"]), Platt.from_dict(d["platt"]), float(d["threshold"]), d.get("metrics", {})
        )


@dataclass
class CaseLibrary:
    """L2-normalised embeddings of labelled reference cases, for case-based explanation."""

    emb: np.ndarray  # [K, D] float32, unit norm
    labels: np.ndarray  # [K] 0/1
    meta: list[dict[str, Any]]  # per case: dataset, site, grade, thumb (relative path)

    def nearest(self, emb: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
        q = emb / (np.linalg.norm(emb) + 1e-8)
        sims = self.emb @ q
        idx = np.argsort(-sims)[:k]
        return [{**self.meta[i], "label": int(self.labels[i]), "similarity": float(sims[i])} for i in idx]


@dataclass
class TaskModel:
    task: str
    head: LinearHead
    platt: Platt
    threshold: float
    target_sensitivity: float
    conformal: ClassConditionalConformal
    ood: OODDetector
    reference: np.ndarray
    backbone: str = ""
    gradability: GradabilityHead | None = None
    library: CaseLibrary | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backbone": self.backbone,
            "head": self.head.to_dict(),
            "platt": self.platt.to_dict(),
            "threshold": self.threshold,
            "target_sensitivity": self.target_sensitivity,
            "conformal": self.conformal.to_dict(),
            "ood": self.ood.to_dict(),
            "gradability": self.gradability.to_dict() if self.gradability else None,
            "metrics": self.metrics,
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], reference: np.ndarray, library: CaseLibrary | None = None) -> TaskModel:
        return cls(
            task=d["task"],
            backbone=d.get("backbone", ""),
            head=LinearHead.from_dict(d["head"]),
            platt=Platt.from_dict(d["platt"]),
            threshold=float(d["threshold"]),
            target_sensitivity=float(d["target_sensitivity"]),
            conformal=ClassConditionalConformal.from_dict(d["conformal"]),
            ood=OODDetector.from_dict(d["ood"]),
            gradability=GradabilityHead.from_dict(d["gradability"]) if d.get("gradability") else None,
            reference=reference,
            library=library,
            metrics=d.get("metrics", {}),
            training=d.get("training", {}),
        )


@dataclass
class Bundle:
    root: Path
    manifest: dict[str, Any]
    backbones: dict[str, OnnxBackbone]
    tasks: dict[str, TaskModel]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def is_demo(self) -> bool:
        return bool(self.manifest.get("demo"))

    def backbone_for(self, task: str) -> OnnxBackbone:
        return self.backbones[self.tasks[task].backbone]

    def evidence(self) -> dict[str, Any] | None:
        p = self.root / "evidence.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @classmethod
    def load(cls, root: str | Path, threads: int = 0) -> Bundle:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        backbones = {
            name: OnnxBackbone(root / "backbones" / _dirname(name), BackboneSpec.from_dict(spec), threads=threads)
            for name, spec in manifest["backbones"].items()
        }
        tasks = {}
        for task in manifest["tasks"]:
            d = json.loads((root / "tasks" / f"{task}.json").read_text(encoding="utf-8"))
            ref = np.load(root / "tasks" / f"{task}_ref.npy").astype(np.float32)
            lib = None
            lib_path = root / "tasks" / f"{task}_library.npz"
            if lib_path.exists():
                with np.load(lib_path, allow_pickle=False) as z:
                    lib = CaseLibrary(z["emb"].astype(np.float32), z["labels"], json.loads(str(z["meta"])))
            tasks[task] = TaskModel.from_dict(d, ref, lib)
        return cls(root=root, manifest=manifest, backbones=backbones, tasks=tasks)


def _dirname(backbone_name: str) -> str:
    return backbone_name.replace("@", "-").replace("/", "-")


def backbone_dir(root: str | Path, backbone_name: str) -> Path:
    return Path(root) / "backbones" / _dirname(backbone_name)


def save_bundle(
    root: str | Path,
    specs: dict[str, BackboneSpec],
    tasks: dict[str, TaskModel],
    *,
    datasets: list[str],
    demo: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Write task files and manifest into ``root``. Backbone ONNX files must already be in
    ``backbones/<name>/`` (see :func:`backbone_dir`)."""
    root = Path(root)
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("".join(sorted(s.sha256 for s in specs.values())).encode())
    for name, tm in tasks.items():
        blob = json.dumps(tm.to_dict(), separators=(",", ":"))
        (root / "tasks" / f"{name}.json").write_text(blob, encoding="utf-8")
        np.save(root / "tasks" / f"{name}_ref.npy", tm.reference.astype(np.float16))
        if tm.library is not None:
            np.savez_compressed(
                root / "tasks" / f"{name}_library.npz",
                emb=tm.library.emb.astype(np.float16),
                labels=tm.library.labels.astype(np.int8),
                meta=np.array(json.dumps(tm.library.meta)),
            )
        digest.update(blob.encode())
    manifest = {
        "bundle_id": digest.hexdigest()[:16],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "clinictune_version": __version__,
        "demo": demo,
        "backbones": {name: s.to_dict() for name, s in specs.items()},
        "tasks": {k: {**TASK_LABELS[k], "backbone": tm.backbone} for k, tm in tasks.items()},
        "datasets": datasets,
        "intended_use": DISCLAIMER,
        "notes": notes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
