"""Frozen backbone served through ONNX Runtime.

Contract for every backbone ONNX file: input ``pixel_values`` float32 [B, 3, S, S] (already normalised),
output ``tokens`` float32 [B, N, D] - the final-layer *patch* tokens (no CLS), N = gh * gw.
The image embedding is the mean over tokens. Embedding extraction for training uses this exact class,
so the features a head is trained on are bit-for-bit the features it sees in production.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from clinictune.imaging import Preprocess


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    onnx_file: str
    sha256: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    grid: tuple[int, int]
    dim: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BackboneSpec:
        return cls(
            name=d["name"],
            onnx_file=d["onnx_file"],
            sha256=d["sha256"],
            input_size=int(d["input_size"]),
            mean=tuple(d["mean"]),  # type: ignore[arg-type]
            std=tuple(d["std"]),  # type: ignore[arg-type]
            grid=tuple(d["grid"]),  # type: ignore[arg-type]
            dim=int(d["dim"]),
        )


def sha256_file(path: str | Path) -> str:
    """sha256 of an ONNX file plus its external-weights sidecar (``<file>.data``) when present."""
    h = hashlib.sha256()
    path = Path(path)
    for p in (path, path.with_name(path.name + ".data")):
        if not p.exists():
            continue
        with p.open("rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class IntegrityError(RuntimeError):
    pass


class OnnxBackbone:
    def __init__(self, root: str | Path, spec: BackboneSpec, threads: int = 0, verify: bool = True) -> None:
        path = Path(root) / spec.onnx_file
        if verify and sha256_file(path) != spec.sha256:
            raise IntegrityError(f"{path} does not match the sha256 recorded in the bundle manifest")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.spec = spec
        self.preprocess = Preprocess(spec.input_size, spec.mean, spec.std)

    def tokens(self, pixel_values: np.ndarray) -> np.ndarray:
        (out,) = self.session.run(["tokens"], {"pixel_values": pixel_values.astype(np.float32)})
        return out

    def embed(self, rgbs: Sequence[np.ndarray], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """-> (embeddings [B, D], tokens [B, N, D])."""
        embs, toks = [], []
        for i in range(0, len(rgbs), batch_size):
            batch = np.stack([self.preprocess(x) for x in rgbs[i : i + batch_size]])
            t = self.tokens(batch)
            toks.append(t)
            embs.append(t.mean(axis=1))
        return np.concatenate(embs), np.concatenate(toks)

    def benchmark(self, n: int = 10, batch_size: int = 1) -> dict[str, float]:
        x = (
            np.random.default_rng(0)
            .standard_normal((batch_size, 3, self.spec.input_size, self.spec.input_size))
            .astype(np.float32)
        )
        self.tokens(x)  # warm-up
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.tokens(x)
            times.append((time.perf_counter() - t0) * 1000 / batch_size)
        return {
            "backbone": self.spec.name,
            "ms_per_image_median": float(np.median(times)),
            "ms_per_image_p90": float(np.percentile(times, 90)),
            "batch_size": batch_size,
        }
