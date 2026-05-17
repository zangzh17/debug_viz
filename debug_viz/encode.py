"""Array -> (full_png_bytes, thumb_png_bytes, metadata)."""
from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image

THUMB_MAX = 800
_EPS = 1e-12


def _stats(arr: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        flat = arr.astype(np.float64, copy=False).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size:
            out["min"] = float(finite.min())
            out["mean"] = float(finite.mean())
            out["max"] = float(finite.max())
        else:
            out["min"] = None
            out["mean"] = None
            out["max"] = None
        nan_count = int(np.isnan(flat).sum())
        if nan_count:
            out["nan_pct"] = float(nan_count / flat.size * 100.0)
    except Exception:
        pass
    return out


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _make_thumb(img: Image.Image) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail((THUMB_MAX, THUMB_MAX))
    return thumb


def _norm_2d(arr: np.ndarray, scale: str, clip_pct: tuple[float, float]) -> tuple[np.ndarray, tuple[float, float]]:
    """Return (uint8 grayscale, (lo, hi) used)."""
    a = arr.astype(np.float64, copy=False)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
        norm = np.zeros_like(a, dtype=np.float64)
    else:
        lo = float(np.percentile(finite, clip_pct[0]))
        hi = float(np.percentile(finite, clip_pct[1]))
        if scale == "log":
            # shift so min > 0
            shift = 0.0
            if lo <= 0:
                shift = -lo + 1.0
            a_pos = np.where(np.isfinite(a), a + shift, np.nan)
            lo_l = math.log10(lo + shift) if (lo + shift) > 0 else 0.0
            hi_l = math.log10(hi + shift) if (hi + shift) > 0 else lo_l + 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                a_log = np.log10(np.where(a_pos > 0, a_pos, np.nan))
            if hi_l - lo_l < _EPS:
                norm = np.ones_like(a, dtype=np.float64)
            else:
                norm = (a_log - lo_l) / (hi_l - lo_l)
        else:
            if hi - lo < _EPS:
                norm = np.ones_like(a, dtype=np.float64)
            else:
                norm = (a - lo) / (hi - lo)
    nan_mask = ~np.isfinite(a)
    norm = np.where(nan_mask, 0.0, np.clip(norm, 0.0, 1.0))
    gray = (norm * 255).astype(np.uint8)
    return gray, (lo, hi), nan_mask


def _gray_with_nan_overlay(gray: np.ndarray, nan_mask: np.ndarray) -> Image.Image:
    """Convert grayscale to RGB and paint NaN pixels red."""
    if not nan_mask.any():
        return Image.fromarray(gray, mode="L").convert("RGB")
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[nan_mask] = (255, 0, 0)
    return Image.fromarray(rgb, mode="RGB")


def _mask_to_image(arr: np.ndarray) -> Image.Image:
    """2-valued -> yellow/black."""
    truthy = arr.astype(bool) if arr.dtype != bool else arr
    rgb = np.zeros((*truthy.shape, 3), dtype=np.uint8)
    rgb[truthy] = (255, 220, 0)
    return Image.fromarray(rgb, mode="RGB")


def _is_two_valued(arr: np.ndarray) -> bool:
    if arr.dtype == bool:
        return True
    if not np.issubdtype(arr.dtype, np.integer):
        return False
    try:
        u = np.unique(arr)
    except Exception:
        return False
    return u.size == 2


def _rgb_uint(arr: np.ndarray) -> Image.Image:
    a = arr
    if a.dtype == np.uint8:
        return Image.fromarray(a, mode="RGB")
    if a.dtype == np.uint16:
        a8 = (a / 257).astype(np.uint8)
        return Image.fromarray(a8, mode="RGB")
    # fallback
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < _EPS:
        a8 = np.ones_like(a, dtype=np.uint8) * 255
    else:
        a8 = ((a - lo) / (hi - lo) * 255).astype(np.uint8)
    return Image.fromarray(a8, mode="RGB")


def _rgb_float(arr: np.ndarray) -> Image.Image:
    a = np.clip(arr.astype(np.float64), 0.0, 1.0)
    a8 = (a * 255).astype(np.uint8)
    return Image.fromarray(a8, mode="RGB")


def _degrade(arr: Any) -> tuple[Image.Image, str]:
    note = f"unsupported: type={type(arr).__name__} repr={repr(arr)[:120]}"
    try:
        shape = getattr(arr, "shape", None)
        if shape is not None:
            note += f" shape={shape}"
    except Exception:
        pass
    img = Image.new("RGB", (64, 64), (40, 40, 40))
    return img, note


def encode_array(
    x: Any,
    *,
    scale: str = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
) -> dict[str, Any]:
    """Return dict with full_png, thumb_png (bytes), and metadata fields."""
    note: str | None = None
    used_clip: tuple[float, float] = (0.0, 1.0)

    try:
        arr = np.asarray(x)
    except Exception:
        img, note = _degrade(x)
        return _finalize(img, shape=None, dtype="?", stats={}, scale=scale,
                         clip=used_clip, note=note)

    shape = list(arr.shape) if hasattr(arr, "shape") else None
    dtype = str(arr.dtype) if hasattr(arr, "dtype") else "?"

    try:
        if arr.ndim == 2:
            if _is_two_valued(arr):
                img = _mask_to_image(arr)
            else:
                gray, used_clip, nan_mask = _norm_2d(arr, scale, clip_pct)
                img = _gray_with_nan_overlay(gray, nan_mask)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            if np.issubdtype(arr.dtype, np.floating):
                img = _rgb_float(arr)
            else:
                img = _rgb_uint(arr)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            img = _rgb_uint(arr[..., :3]) if not np.issubdtype(arr.dtype, np.floating) else _rgb_float(arr[..., :3])
            note = f"showing first 3 of {arr.shape[2]} channels"
        else:
            img, note = _degrade(arr)
    except Exception as e:
        img, note = _degrade(arr)
        note = f"encode error: {e}; {note}"

    stats = _stats(arr) if hasattr(arr, "size") else {}
    return _finalize(img, shape=shape, dtype=dtype, stats=stats,
                     scale=scale, clip=used_clip, note=note)


def _finalize(
    img: Image.Image,
    *,
    shape: list[int] | None,
    dtype: str,
    stats: dict[str, Any],
    scale: str,
    clip: tuple[float, float],
    note: str | None,
) -> dict[str, Any]:
    full_png = _to_png_bytes(img)
    thumb = _make_thumb(img)
    thumb_png = _to_png_bytes(thumb)
    return {
        "full_png": full_png,
        "thumb_png": thumb_png,
        "shape": shape,
        "dtype": dtype,
        "stats": stats,
        "scale": scale,
        "clip": [float(clip[0]), float(clip[1])],
        "note": note,
    }
