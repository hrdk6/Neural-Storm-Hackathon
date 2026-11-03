"""Image decoding and backbone preprocessing, shared by embedding extraction and serving.

Using one code path for both is deliberate: train/serve preprocessing skew is a classic silent failure.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Refuse decompression bombs: 80 MP is far above any CXR / fundus capture.
Image.MAX_IMAGE_PIXELS = 80_000_000


class ImageDecodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Preprocess:
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> float32 3xSxS normalised tensor."""
        img = Image.fromarray(rgb).resize((self.size, self.size), Image.Resampling.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1))


def _from_dicom(data: bytes) -> np.ndarray:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImageDecodeError("DICOM support needs the 'dicom' extra: uv sync --extra dicom") from exc
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    gray = arr.astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def decode(data: bytes) -> np.ndarray:
    """Bytes (PNG/JPEG/TIFF/BMP/DICOM) -> uint8 HxWx3 RGB, EXIF orientation applied."""
    if len(data) > 132 and data[128:132] == b"DICM":
        return _from_dicom(data)
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("I;16", "I;16B", "I"):
                arr = np.asarray(im, dtype=np.float32)
                arr = (arr - arr.min()) / max(float(np.ptp(arr)), 1e-6) * 255
                im = Image.fromarray(arr.astype(np.uint8))
            rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"could not decode image: {exc}") from exc
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 16:
        raise ImageDecodeError(f"image too small or malformed: shape {rgb.shape}")
    return rgb


def load(path: str | Path) -> np.ndarray:
    return decode(Path(path).read_bytes())


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)
