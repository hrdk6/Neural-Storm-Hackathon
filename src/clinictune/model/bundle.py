"""Model bundle: one directory holding everything needed to serve.

    <bundle>/manifest.json          bundle id, backbone spec, tasks, intended use, licences
    <bundle>/backbone.onnx[.data]   frozen backbone (sha256-checked at load)
    <bundle>/tasks/<task>.json      head, source calibration, threshold, conformal, OOD, metrics
    <bundle>/tasks/<task>_ref.npy   subsample of training embeddings (for shift tests at onboarding)

Bundles contain model weights and derived data - they are never committed to git.
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
class TaskModel:
    task: str
    head: LinearHead
    platt: Platt
    threshold: float
    target_sensitivity: float
    conformal: ClassConditionalConformal
    ood: OODDetector
    reference: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "head": self.head.to_dict(),
            "platt": self.platt.to_dict(),
            "threshold": self.threshold,
            "target_sensitivity": self.target_sensitivity,
            "conformal": self.conformal.to_dict(),
            "ood": self.ood.to_dict(),
            "metrics": self.metrics,
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], reference: np.ndarray) -> TaskModel:
        return cls(
            task=d["task"],
            head=LinearHead.from_dict(d["head"]),
            platt=Platt.from_dict(d["platt"]),
            threshold=float(d["threshold"]),
            target_sensitivity=float(d["target_sensitivity"]),
            conformal=ClassConditionalConformal.from_dict(d["conformal"]),
            ood=OODDetector.from_dict(d["ood"]),
            reference=reference,
            metrics=d.get("metrics", {}),
            training=d.get("training", {}),
        )


@dataclass
class Bundle:
    root: Path
    manifest: dict[str, Any]
    backbone: OnnxBackbone
    tasks: dict[str, TaskModel]

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def is_demo(self) -> bool:
        return bool(self.manifest.get("demo"))

    @classmethod
    def load(cls, root: str | Path, threads: int = 0) -> Bundle:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        spec = BackboneSpec.from_dict(manifest["backbone"])
        backbone = OnnxBackbone(root, spec, threads=threads)
        tasks = {}
        for task in manifest["tasks"]:
            d = json.loads((root / "tasks" / f"{task}.json").read_text(encoding="utf-8"))
            ref = np.load(root / "tasks" / f"{task}_ref.npy").astype(np.float32)
            tasks[task] = TaskModel.from_dict(d, ref)
        return cls(root=root, manifest=manifest, backbone=backbone, tasks=tasks)


def save_bundle(
    root: str | Path,
    spec: BackboneSpec,
    tasks: dict[str, TaskModel],
    *,
    datasets: list[str],
    demo: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Write task files and manifest into ``root`` (the backbone ONNX must already be there)."""
    root = Path(root)
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(spec.sha256.encode())
    for name, tm in tasks.items():
        blob = json.dumps(tm.to_dict(), separators=(",", ":"))
        (root / "tasks" / f"{name}.json").write_text(blob, encoding="utf-8")
        np.save(root / "tasks" / f"{name}_ref.npy", tm.reference.astype(np.float16))
        digest.update(blob.encode())
    manifest = {
        "bundle_id": digest.hexdigest()[:16],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "clinictune_version": __version__,
        "demo": demo,
        "backbone": spec.to_dict(),
        "tasks": {k: TASK_LABELS[k] for k in tasks},
        "datasets": datasets,
        "intended_use": DISCLAIMER,
        "notes": notes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
