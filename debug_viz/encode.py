"""Array -> image bytes + metadata. Supports kind hints and multi-band output."""
from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image

THUMB_MAX = 400
DEFAULT_FMT = "jpeg"        # 20x faster than WebP on incompressible 6kx6k data
JPEG_QUALITY = 95            # near-lossless for grayscale science viz
_EPS = 1e-12

KINDS = ("auto", "gray", "mono", "rgb", "raw", "multi", "mask")


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
            out["min"] = out["mean"] = out["max"] = None
        nan_count = int(np.isnan(flat).sum())
        if nan_count:
            out["nan_pct"] = float(nan_count / flat.size * 100.0)
    except Exception:
        pass
    return out


def _resolved_fmt(fmt: str, lossless: bool) -> str:
    """Mask / NaN overlay -> PNG (exact red/yellow); otherwise stick with fmt."""
    if lossless:
        return "png"
    return fmt


def _encode_image(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=False)
    elif fmt == "png":
        img.save(buf, format="PNG", compress_level=1)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=92, method=2)
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
    return buf.getvalue()


def _downsample(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Stride-sample to bring largest dim <= max_dim. Cheap O(n/s²)."""
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[:2]
    s = max(1, math.ceil(max(h, w) / max_dim))
    if s == 1:
        return arr
    return arr[::s, ::s, ...] if arr.ndim == 3 else arr[::s, ::s]


_PCT_SAMPLE_LIMIT = 1_000_000  # subsample threshold for percentile (huge speedup)
_PCT_RNG = np.random.default_rng(0)


def _compute_clip(arr: np.ndarray, clip_pct: tuple[float, float]) -> tuple[float, float]:
    flat = arr.ravel()
    if np.issubdtype(flat.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
    else:
        finite = flat
    if finite.size == 0:
        return 0.0, 1.0
    # Random subsample for huge arrays — percentile estimates are robust at 1M
    # samples (1% relative error << visualization tolerance).
    if finite.size > _PCT_SAMPLE_LIMIT:
        idx = _PCT_RNG.integers(0, finite.size, _PCT_SAMPLE_LIMIT)
        finite = finite[idx]
    lo, hi = np.percentile(finite, clip_pct)
    return float(lo), float(hi)


def _normalize_with_clip(
    arr: np.ndarray, scale: str, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray]:
    # Stay in float32 — half the memory bandwidth vs float64 for huge arrays.
    if arr.dtype != np.float32:
        a = arr.astype(np.float32, copy=False)
    else:
        a = arr
    is_float = np.issubdtype(a.dtype, np.floating)
    nan_mask = ~np.isfinite(a) if is_float else np.zeros(a.shape, dtype=bool)

    if scale == "log":
        shift = -lo + 1.0 if lo <= 0 else 0.0
        lo_l = math.log10(lo + shift) if (lo + shift) > 0 else 0.0
        hi_l = math.log10(hi + shift) if (hi + shift) > 0 else lo_l + 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            shifted = a + shift if shift else a
            bad = ~(shifted > 0)
            a_log = np.log10(np.where(bad, np.nan, shifted))
        denom = hi_l - lo_l
        if denom < _EPS:
            norm = np.ones_like(a, dtype=np.float32)
        else:
            norm = (a_log - lo_l) * (1.0 / denom)
        nan_mask = nan_mask | ~np.isfinite(norm)
    else:
        denom = hi - lo
        if denom < _EPS:
            norm = np.ones_like(a, dtype=np.float32)
        else:
            # Single-pass: (a - lo) / (hi - lo). In-place to avoid extra alloc.
            norm = (a - lo) * (1.0 / denom)

    np.clip(norm, 0.0, 1.0, out=norm)
    if nan_mask.any():
        norm[nan_mask] = 0.0
    return (norm * 255).astype(np.uint8), nan_mask


def _gray_to_rgb(gray: np.ndarray, nan_mask: np.ndarray) -> Image.Image:
    if nan_mask.any():
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[nan_mask] = (255, 0, 0)
        return Image.fromarray(rgb, mode="RGB")
    return Image.fromarray(gray, mode="L").convert("RGB")


def _mask_to_image(arr: np.ndarray) -> Image.Image:
    truthy = arr.astype(bool)
    rgb = np.zeros((*truthy.shape, 3), dtype=np.uint8)
    rgb[truthy] = (255, 220, 0)
    return Image.fromarray(rgb, mode="RGB")


def _is_two_valued(arr: np.ndarray) -> bool:
    if arr.dtype == bool:
        return True
    if not np.issubdtype(arr.dtype, np.integer):
        return False
    try:
        return np.unique(arr).size == 2
    except Exception:
        return False


def _render_2d(arr: np.ndarray, scale: str, lo: float, hi: float) -> Image.Image:
    gray, nan_mask = _normalize_with_clip(arr, scale, lo, hi)
    return _gray_to_rgb(gray, nan_mask)


def _render_rgb(arr: np.ndarray) -> Image.Image:
    if np.issubdtype(arr.dtype, np.floating):
        a8 = (np.clip(arr.astype(np.float64), 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(a8, mode="RGB")
    if arr.dtype == np.uint8:
        return Image.fromarray(arr, mode="RGB")
    if arr.dtype == np.uint16:
        return Image.fromarray((arr / 257).astype(np.uint8), mode="RGB")
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < _EPS:
        return Image.fromarray((np.ones_like(a) * 255).astype(np.uint8), mode="RGB")
    return Image.fromarray(((a - lo) / (hi - lo) * 255).astype(np.uint8), mode="RGB")


def _render_multi_composite(
    arr: np.ndarray, bands: tuple[int, ...], scale: str, clip_pct: tuple[float, float],
) -> Image.Image:
    """Build RGB composite from selected channels of a multi-band array.

    Each band normalized independently (per-band percentile)."""
    n = arr.shape[-1]
    chs: list[np.ndarray] = []
    for i in range(3):
        idx = bands[i] if i < len(bands) else bands[-1]
        idx = max(0, min(n - 1, int(idx)))
        band = arr[..., idx]
        lo, hi = _compute_clip(band, clip_pct)
        gray, _ = _normalize_with_clip(band, scale, lo, hi)
        chs.append(gray)
    rgb = np.stack(chs, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _degrade(x: Any) -> tuple[Image.Image, str]:
    note = f"unsupported: type={type(x).__name__} repr={repr(x)[:120]}"
    try:
        shape = getattr(x, "shape", None)
        if shape is not None:
            note += f" shape={shape}"
    except Exception:
        pass
    return Image.new("RGB", (64, 64), (40, 40, 40)), note


def _resolve_kind(arr: np.ndarray, kind: str) -> str:
    if kind != "auto":
        return kind
    if arr.ndim == 2 and _is_two_valued(arr):
        return "mask"
    if arr.ndim == 2:
        return "gray"
    if arr.ndim == 3 and arr.shape[2] == 3:
        return "rgb"
    if arr.ndim == 3 and arr.shape[2] > 3:
        return "multi"
    return "gray"


def _encode_pair(img: Image.Image, thumb_img: Image.Image, fmt: str) -> tuple[bytes, bytes]:
    return _encode_image(img, fmt), _encode_image(thumb_img, fmt)


def encode_array(
    x: Any,
    *,
    kind: str = "auto",
    bands: tuple[int, ...] | None = None,
    scale: str = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
    fmt: str = DEFAULT_FMT,
) -> dict[str, Any]:
    """Encode array. Returns dict with main_full/main_thumb, optional 'bands' list."""
    if kind not in KINDS:
        kind = "auto"

    note: str | None = None

    try:
        arr = np.asarray(x)
    except Exception:
        img, note = _degrade(x)
        return _make_result(img, img, kind="unknown", shape=None, dtype="?",
                            stats={}, scale=scale, clip=(0.0, 1.0), note=note,
                            fmt=fmt, bands_data=None)

    shape = list(arr.shape) if hasattr(arr, "shape") else None
    dtype = str(arr.dtype) if hasattr(arr, "dtype") else "?"
    stats = _stats(arr)
    resolved = _resolve_kind(arr, kind)

    if resolved == "raw":
        note = "raw image (no demosaic)"

    # Lossless for mask + NaN-overlay (preserves exact red/yellow markers)
    lossless = False

    try:
        if resolved == "mask":
            full_img = _mask_to_image(arr)
            thumb_img = _mask_to_image(_downsample(arr, THUMB_MAX))
            clip_used = (0.0, 1.0)
            lossless = True
        elif resolved in ("gray", "mono", "raw"):
            if arr.ndim != 2:
                raise ValueError(f"kind={resolved!r} expects 2-D array, got ndim={arr.ndim}")
            lo, hi = _compute_clip(arr, clip_pct)
            full_img = _render_2d(arr, scale, lo, hi)
            thumb_img = _render_2d(_downsample(arr, THUMB_MAX), scale, lo, hi)
            clip_used = (lo, hi)
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                lossless = True
        elif resolved == "rgb":
            if arr.ndim != 3 or arr.shape[2] < 3:
                raise ValueError(f"kind='rgb' expects (H,W,>=3), got shape={arr.shape}")
            sub = arr[..., :3]
            full_img = _render_rgb(sub)
            thumb_img = _render_rgb(_downsample(sub, THUMB_MAX))
            clip_used = (0.0, 1.0)
        elif resolved == "multi":
            if arr.ndim != 3:
                raise ValueError(f"kind='multi' expects 3-D array, got ndim={arr.ndim}")
            n_ch = arr.shape[2]
            sel = tuple(bands) if bands else tuple(range(min(3, n_ch)))
            sel = tuple(max(0, min(n_ch - 1, int(b))) for b in sel)
            full_img = _render_multi_composite(arr, sel, scale, clip_pct)
            thumb_img = _render_multi_composite(_downsample(arr, THUMB_MAX), sel, scale, clip_pct)
            clip_used = (0.0, 1.0)
            note = f"{n_ch} channels, composite=R:ch{sel[0]} G:ch{sel[1] if len(sel)>1 else sel[0]} B:ch{sel[2] if len(sel)>2 else sel[0]}"
        else:
            full_img, note = _degrade(arr)
            thumb_img = full_img
            clip_used = (0.0, 1.0)
    except Exception as e:
        full_img, n = _degrade(arr)
        note = f"encode error: {e}; {n}"
        thumb_img = full_img
        clip_used = (0.0, 1.0)

    # If multi has any NaN, escalate the whole frame to lossless (rare case).
    if resolved == "multi" and arr.ndim == 3 and not np.isfinite(arr).all():
        lossless = True
    eff_fmt = _resolved_fmt(fmt, lossless)

    # Per-band rendering for multi (all use same fmt as composite)
    bands_data: list[dict[str, Any]] | None = None
    if resolved == "multi" and arr.ndim == 3:
        bands_data = []
        for i in range(arr.shape[2]):
            band = arr[..., i]
            blo, bhi = _compute_clip(band, clip_pct)
            try:
                bfull = _render_2d(band, scale, blo, bhi)
                bthumb = _render_2d(_downsample(band, THUMB_MAX), scale, blo, bhi)
                bfull_b, bthumb_b = _encode_pair(bfull, bthumb, eff_fmt)
                bands_data.append({
                    "label": f"ch {i}",
                    "stats": _stats(band),
                    "clip": [float(blo), float(bhi)],
                    "full": bfull_b,
                    "thumb": bthumb_b,
                })
            except Exception:
                pass

    return _make_result(full_img, thumb_img, kind=resolved, shape=shape, dtype=dtype,
                        stats=stats, scale=scale, clip=clip_used, note=note,
                        fmt=eff_fmt, bands_data=bands_data)


def _make_result(
    full_img: Image.Image,
    thumb_img: Image.Image,
    *,
    kind: str,
    shape: list[int] | None,
    dtype: str,
    stats: dict[str, Any],
    scale: str,
    clip: tuple[float, float],
    note: str | None,
    fmt: str,
    bands_data: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    main_full, main_thumb = _encode_pair(full_img, thumb_img, fmt)
    result = {
        "fmt": fmt,
        "kind": kind,
        "main_full": main_full,
        "main_thumb": main_thumb,
        "shape": shape,
        "dtype": dtype,
        "stats": stats,
        "scale": scale,
        "clip": [float(clip[0]), float(clip[1])],
        "note": note,
    }
    if bands_data:
        result["bands"] = bands_data
    return result
