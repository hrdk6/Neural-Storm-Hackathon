"""Read an onboarding pack (zip of images + labels.csv) safely and run the full onboarding.

The zip is read in memory entry by entry - nothing is extracted to disk, so path traversal is impossible,
and total uncompressed size / entry count are capped against zip bombs.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from clinictune import imaging, quality
from clinictune.model.bundle import Bundle
from clinictune.site.onboarding import SiteCalibration, onboard_site


class PackError(ValueError):
    pass


def read_pack(data: bytes, max_images: int, max_uncompressed: int) -> list[tuple[str, bytes, int, str]]:
    """-> [(filename, image bytes, label, patient_id)]"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise PackError("upload is not a valid zip file") from exc
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > max_images + 10:
        raise PackError(f"too many files in pack (max {max_images} images)")
    if sum(i.file_size for i in infos) > max_uncompressed:
        raise PackError("pack is too large when uncompressed")
    by_name = {PurePosixPath(i.filename).name: i for i in infos}
    labels_info = by_name.get("labels.csv")
    if labels_info is None:
        raise PackError("labels.csv not found in pack (columns: filename,label[,patient_id])")
    text = zf.read(labels_info).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or not {"filename", "label"} <= set(rows[0]):
        raise PackError("labels.csv needs columns: filename,label[,patient_id]")
    out = []
    for i, row in enumerate(rows, start=2):
        name = PurePosixPath(row["filename"].strip()).name
        if name not in by_name:
            raise PackError(f"labels.csv line {i}: {name!r} not found in pack")
        label = row["label"].strip().lower()
        if label not in {"0", "1", "positive", "negative", "refer", "normal"}:
            raise PackError(f"labels.csv line {i}: label must be 0/1, got {row['label']!r}")
        y = int(label in {"1", "positive", "refer"})
        pid = (row.get("patient_id") or "").strip() or name
        out.append((name, zf.read(by_name[name]), y, pid))
    if len(out) > max_images:
        raise PackError(f"too many images ({len(out)} > {max_images})")
    return out


def run_onboarding(
    bundle: Bundle,
    site_id: str,
    task: str,
    pack: list[tuple[str, bytes, int, str]],
    progress: Callable[[float, str], None] = lambda p, m: None,
) -> tuple[SiteCalibration, dict[str, Any]]:
    tm = bundle.tasks[task]
    kept_rgb, kept_names, y, groups = [], [], [], []
    rejected: list[dict[str, Any]] = []
    for i, (name, data, label, pid) in enumerate(pack):
        try:
            rgb = imaging.decode(data)
        except imaging.ImageDecodeError as exc:
            rejected.append({"file": name, "reasons": [str(exc)]})
            continue
        gate = quality.check(task, rgb)
        if not gate.passed:
            rejected.append({"file": name, "reasons": [c.name for c in gate.failed]})
            continue
        kept_rgb.append(rgb)
        kept_names.append(name)
        y.append(label)
        groups.append(pid)
        if i % 10 == 0:
            progress(0.3 * i / len(pack), f"quality gate {i}/{len(pack)}")
    backbone = bundle.backbone_for(task)
    embs = []
    for start in range(0, len(kept_rgb), 8):
        e, _ = backbone.embed(kept_rgb[start : start + 8])
        embs.append(e)
        done = min(start + 8, len(kept_rgb))
        progress(0.3 + 0.6 * done / max(len(kept_rgb), 1), f"embedding {done}/{len(kept_rgb)}")
    if not embs:
        raise PackError("no image passed the quality gate")
    emb = np.concatenate(embs)
    y_arr, g_arr = np.array(y), np.array(groups)
    if tm.gradability is not None:  # learned gate: drop images the quality model calls ungradable
        ok = tm.gradability.p_ungradable(emb) < tm.gradability.threshold
        for i in np.flatnonzero(~ok):
            rejected.append({"file": kept_names[i], "reasons": ["learned_gradability"]})
        emb, y_arr, g_arr = emb[ok], y_arr[ok], g_arr[ok]
        kept_rgb = [r for r, keep in zip(kept_rgb, ok, strict=True) if keep]
    gate_summary = {
        "submitted": len(pack),
        "passed": len(kept_rgb),
        "pass_rate": len(kept_rgb) / len(pack),
        "rejected": rejected[:50],
    }
    progress(0.92, "recalibrating")
    return onboard_site(tm, site_id, bundle.bundle_id, emb, y_arr, g_arr, gate_summary=gate_summary)
