"""A tiny random-projection patch encoder, written directly as an ONNX graph.

It exists so the full pipeline (extraction -> heads -> calibration -> conformal -> serving) can run and be
tested without downloading gated foundation-model weights. It is NOT a medical model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from clinictune.embed.backbone import BackboneSpec, sha256_file


def build_demo_backbone(
    out_dir: str | Path, size: int = 224, patch: int = 16, dim: int = 192, seed: int = 0
) -> BackboneSpec:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = size // patch
    pdim = 3 * patch * patch
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((pdim, dim)) / np.sqrt(pdim) * 2.0).astype(np.float32)
    b = (rng.standard_normal(dim) * 0.1).astype(np.float32)

    inits = [
        numpy_helper.from_array(np.array([-1, 3, g, patch, g, patch], dtype=np.int64), "shape6"),
        numpy_helper.from_array(np.array([-1, g * g, pdim], dtype=np.int64), "shape3"),
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(b, "b"),
    ]
    nodes = [
        helper.make_node("Reshape", ["pixel_values", "shape6"], ["r6"]),
        helper.make_node("Transpose", ["r6"], ["t6"], perm=[0, 2, 4, 1, 3, 5]),
        helper.make_node("Reshape", ["t6", "shape3"], ["patches"]),
        helper.make_node("MatMul", ["patches", "W"], ["proj"]),
        helper.make_node("Add", ["proj", "b"], ["pre"]),
        helper.make_node("Tanh", ["pre"], ["tokens"]),
    ]
    graph = helper.make_graph(
        nodes,
        "demo_randproj",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("tokens", TensorProto.FLOAT, ["batch", g * g, dim])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="clinictune")
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = out_dir / "backbone.onnx"
    onnx.save(model, str(path))
    return BackboneSpec(
        name="demo-randproj",
        onnx_file="backbone.onnx",
        sha256=sha256_file(path),
        input_size=size,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        grid=(g, g),
        dim=dim,
    )
