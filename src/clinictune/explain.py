"""Heatmaps from exact per-patch logit contributions (see :mod:`clinictune.model.heads`).

Red = patches pushing towards "refer", blue = towards "low risk". Because the decomposition is exact, the
heatmap explains *this* linear head - it does not claim to localise disease, and the UI says so.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from clinictune.model.heads import LinearHead


def contribution_map(head: LinearHead, tokens: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    return head.token_contributions(tokens).reshape(grid)


def overlay_png(rgb: np.ndarray, cmap_grid: np.ndarray, max_side: int = 512, alpha: float = 0.45) -> bytes:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    base = Image.fromarray(rgb).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    lim = float(np.percentile(np.abs(cmap_grid), 98)) or 1.0
    v = np.clip(cmap_grid / lim, -1, 1)
    heat = np.zeros((*v.shape, 4), dtype=np.float32)
    heat[..., 0] = np.clip(v, 0, 1)  # red: towards refer
    heat[..., 2] = np.clip(-v, 0, 1)  # blue: towards low risk
    heat[..., 3] = np.abs(v) * alpha
    heat_img = Image.fromarray((heat * 255).astype(np.uint8), "RGBA").resize(size, Image.Resampling.BICUBIC)
    out = base.convert("RGBA")
    out.alpha_composite(heat_img)
    buf = io.BytesIO()
    out.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def content_mask(rgb: np.ndarray, grid: tuple[int, int], min_level: float = 12.0) -> np.ndarray:
    """True for patches that contain image content (not the black surround of a fundus photo or film edge)."""
    gray = Image.fromarray(rgb).convert("L").resize((grid[1], grid[0]), Image.Resampling.BOX)
    return np.asarray(gray, dtype=np.float32) > min_level


def top_regions(grid: np.ndarray, k: int = 3, valid: np.ndarray | None = None) -> list[dict[str, float]]:
    """The k patches pushing hardest towards 'refer', as relative boxes (x, y, w, h in 0..1) for the UI.

    ``valid`` restricts the boxes to patches with image content; the heatmap itself stays exact and unmasked.
    """
    gh, gw = grid.shape
    flat = grid.ravel()
    lim = float(np.abs(flat).max()) or 1.0
    order = [i for i in np.argsort(-flat) if valid is None or valid.ravel()[i]]
    out = []
    for i in order[:k]:
        if flat[i] <= 0:
            break
        r, c = divmod(int(i), gw)
        out.append({"x": c / gw, "y": r / gh, "w": 1 / gw, "h": 1 / gh, "weight": float(flat[i] / lim)})
    return out


def to_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()
