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


# ----------------------------------------------------------------------------- task-specific model views

Region = tuple[int, int, int]  # (left, top, side) of the model's square view, in original-image pixels


def fundus_field(rgb: np.ndarray, threshold: float = 15.0) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the retinal field of view, or None if none is found."""
    from scipy.ndimage import binary_fill_holes

    h, w = rgb.shape[:2]
    small = Image.fromarray(rgb)
    small.thumbnail((512, 512), Image.Resampling.BILINEAR)
    gray = to_gray(np.asarray(small))
    fov = binary_fill_holes(gray > threshold)
    if fov.mean() < 0.1:
        return None
    rows, cols = np.flatnonzero(fov.mean(axis=1) > 0.02), np.flatnonzero(fov.mean(axis=0) > 0.02)
    if not len(rows) or not len(cols):
        return None
    sy, sx = h / gray.shape[0], w / gray.shape[1]
    return int(cols[0] * sx), int(rows[0] * sy), int((cols[-1] + 1) * sx), int((rows[-1] + 1) * sy)


def standardize_fundus(rgb: np.ndarray, out_size: int = 512) -> tuple[np.ndarray, Region]:
    """Put every camera's retinal field into the same frame: crop to the field, centre it in a square,
    resize, and black out everything outside the inscribed circle.

    Without this, the size, position and shape of the field (which reveal the camera) can stand in for the
    diagnosis: in APTOS, field geometry alone predicts referable DR with ROC-AUC 0.85.
    Returns the standardised image and where its square sits in the original image.
    """
    h, w = rgb.shape[:2]
    box = fundus_field(rgb)
    x0, y0, x1, y1 = box if box else (0, 0, w, h)
    side = max(x1 - x0, y1 - y0)
    left, top = x0 - (side - (x1 - x0)) // 2, y0 - (side - (y1 - y0)) // 2
    canvas = Image.new("RGB", (side, side))
    canvas.paste(Image.fromarray(rgb), (-left, -top))
    size = out_size  # always the same canonical frame, whatever the camera's resolution
    out = np.asarray(canvas.resize((size, size), Image.Resampling.BILINEAR)).copy()
    yy, xx = np.mgrid[0:size, 0:size]
    r = size / 2
    out[(yy - r + 0.5) ** 2 + (xx - r + 0.5) ** 2 > (0.99 * r) ** 2] = 0
    return out, (left, top, side)


# task -> model view used at training time. The name carries every parameter ("fundus_field:512") and is
# stored in the bundle and in cache file names, so a changed view can never be mixed with stale embeddings.
PREPROCESS = {"dr": "fundus_field:512"}


def view_tag(preprocess: str) -> str:
    """File-name-safe form of a view name."""
    return preprocess.replace(":", "-")


def model_view(preprocess: str, rgb: np.ndarray) -> tuple[np.ndarray, Region | None]:
    """The image exactly as the model sees it for a task, plus where it sits in the original (or None)."""
    name, _, arg = preprocess.partition(":")
    if name == "fundus_field":
        return standardize_fundus(rgb, out_size=int(arg or 512))
    if name:
        raise ValueError(f"unknown model view {preprocess!r}")
    return rgb, None


def view_transform(preprocess: str):  # type: ignore[no-untyped-def]
    """``model_view`` as a plain rgb -> rgb function, for embedding extraction."""
    if not preprocess:
        return None

    def fn(rgb: np.ndarray) -> np.ndarray:
        return model_view(preprocess, rgb)[0]

    fn.__name__ = f"view_{view_tag(preprocess)}"
    return fn
