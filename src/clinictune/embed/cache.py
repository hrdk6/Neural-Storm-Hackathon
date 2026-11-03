"""Extract embeddings once and cache them to disk (``.npz``), keyed by backbone hash and file identity."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from clinictune import imaging
from clinictune.data.manifest import Record
from clinictune.embed.backbone import OnnxBackbone

log = logging.getLogger(__name__)


def _key(path: str) -> str:
    st = os.stat(path)
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def extract(
    records: Sequence[Record],
    backbone: OnnxBackbone,
    cache_path: str | Path,
    batch_size: int = 8,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return embeddings [len(records), D] in record order, reusing any cached rows.

    ``transform`` (e.g. border crop / lung masking for shortcut tests) is applied to the decoded RGB image;
    use a different cache_path per transform.
    """
    cache_path = Path(cache_path)
    keys = [_key(r.image_path) for r in records]
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        # Context manager: an open NpzFile keeps a handle that blocks the atomic replace below on Windows.
        with np.load(cache_path, allow_pickle=False) as z:
            same_backbone = str(z["backbone_sha256"]) == backbone.spec.sha256
            if same_backbone:
                cached = dict(zip(z["keys"].tolist(), z["emb"], strict=True))
        if not same_backbone:
            log.warning("cache %s was built with a different backbone; re-extracting", cache_path)
    todo = [i for i, k in enumerate(keys) if k not in cached]
    log.info("embeddings: %d cached, %d to extract", len(records) - len(todo), len(todo))
    for start in range(0, len(todo), 64):
        chunk = todo[start : start + 64]
        rgbs = []
        for i in chunk:
            rgb = imaging.load(records[i].image_path)
            rgbs.append(transform(rgb) if transform else rgb)
        emb, _ = backbone.embed(rgbs, batch_size=batch_size)
        for i, e in zip(chunk, emb, strict=True):
            cached[keys[i]] = e
        log.info("extracted %d/%d", min(start + 64, len(todo)), len(todo))
    out = np.stack([cached[k] for k in keys]).astype(np.float32)
    if todo or not cache_path.exists():
        # Persist the union of old and new rows, so extracting a subset never drops other cached images.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        all_keys = list(cached)
        np.savez_compressed(
            tmp,
            keys=np.array(all_keys),
            emb=np.stack([cached[k] for k in all_keys]).astype(np.float32),
            backbone_sha256=np.array(backbone.spec.sha256),
        )
        tmp.replace(cache_path)
    return out
