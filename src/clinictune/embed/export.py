"""One-off export of a real foundation backbone to the ClinicTune ONNX contract.

Needs the ``export`` extra (torch + transformers):  ``uv sync --extra export``.
MedSigLIP is gated: accept the HAI-DEF terms on Hugging Face and ``hf auth login`` first.
"""

from __future__ import annotations

from pathlib import Path

from clinictune.embed.backbone import BackboneSpec, sha256_file

PRESETS: dict[str, dict] = {
    "medsiglip-448": {
        "hf_id": "google/medsiglip-448",
        "input_size": 448,
        "patch": 14,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "drop_cls": False,
    },
    "dinov2-small": {
        "hf_id": "facebook/dinov2-small",
        "input_size": 224,
        "patch": 14,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "drop_cls": True,
    },
}


def export_backbone(name: str, out_dir: str | Path, opset: int = 18) -> BackboneSpec:
    import torch
    from transformers import AutoModel

    preset = PRESETS[name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(preset["hf_id"])
    vision = getattr(model, "vision_model", model).eval()
    drop_cls = preset["drop_cls"]

    class Tokens(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = vision

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            h = self.vision(pixel_values=pixel_values).last_hidden_state
            return h[:, 1:, :] if drop_cls else h

    size = preset["input_size"]
    dummy = torch.randn(2, 3, size, size)
    with torch.no_grad():
        ref = Tokens()(dummy)
        program = torch.onnx.export(
            Tokens(),
            (dummy,),
            input_names=["pixel_values"],
            output_names=["tokens"],
            dynamic_shapes={"pixel_values": {0: torch.export.Dim.DYNAMIC}},
            opset_version=opset,
            dynamo=True,
            external_data=True,
        )
    path = out_dir / "backbone.onnx"
    program.save(str(path), external_data=True)

    grid = size // preset["patch"]
    if ref.shape[1] != grid * grid:
        raise RuntimeError(f"expected {grid * grid} patch tokens, model produced {ref.shape[1]}")
    _verify(path, dummy.numpy(), ref.numpy())
    return BackboneSpec(
        name=name,
        onnx_file="backbone.onnx",
        sha256=sha256_file(path),
        input_size=size,
        mean=preset["mean"],
        std=preset["std"],
        grid=(grid, grid),
        dim=int(ref.shape[-1]),
    )


def _verify(path: Path, x, ref) -> None:  # type: ignore[no-untyped-def]
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(["tokens"], {"pixel_values": x})
    err = float(np.abs(out - ref).max())
    if err > 1e-2:
        raise RuntimeError(f"ONNX output differs from PyTorch by {err:.4f}")
