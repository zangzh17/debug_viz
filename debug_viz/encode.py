"""Array -> PIL image + metadata.

This module is the pure rendering core. It accepts a numpy array + display
parameters (scale / clip_pct / bands / wavelength(s)) and returns a dict with
a PIL.Image (the "main view"), per-band PIL.Images for multi data, and
descriptive metadata. Byte encoding + DZI tile generation are downstream
(see dzi.write_dzi)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from .spectral import (
    composite_multi_by_wavelengths,
    tint_gray_by_wavelength,
)

_EPS = 1e-12

KINDS = ("auto", "gray", "mono", "rgb", "raw", "multi", "mask")


def stats(arr: np.ndarray) -> dict[str, Any]:
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


_PCT_SAMPLE_LIMIT = 1_000_000
_PCT_RNG = np.random.default_rng(0)


def _compute_clip(arr: np.ndarray, clip_pct: tuple[float, float]) -> tuple[float, float]:
    flat = arr.ravel()
    if np.issubdtype(flat.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
    else:
        finite = flat
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > _PCT_SAMPLE_LIMIT:
        idx = _PCT_RNG.integers(0, finite.size, _PCT_SAMPLE_LIMIT)
        finite = finite[idx]
    lo, hi = np.percentile(finite, clip_pct)
    return float(lo), float(hi)


def _normalize_with_clip(
    arr: np.ndarray, scale: str, lo: float, hi: float,
    log_dr: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize arr into [0,1] float32 with given scale + clip. Returns (norm, nan_mask).

    `log_dr`: when scale=='log', cap the displayed dynamic range to this many
    orders of magnitude — values below `hi / 10**log_dr` are clipped to black.
    None = no cap (use the full percentile range)."""
    if arr.dtype != np.float32:
        a = arr.astype(np.float32, copy=False)
    else:
        a = arr
    is_float = np.issubdtype(a.dtype, np.floating)
    nan_mask = ~np.isfinite(a) if is_float else np.zeros(a.shape, dtype=bool)

    # Constant arrays: if the constant is zero (or below), render as black;
    # otherwise render as full intensity. This matters for multi-band data
    # where dead channels would otherwise pollute a wavelength composite.
    if scale == "log":
        shift = -lo + 1.0 if lo <= 0 else 0.0
        lo_l = math.log10(lo + shift) if (lo + shift) > 0 else 0.0
        hi_l = math.log10(hi + shift) if (hi + shift) > 0 else lo_l + 1.0
        # Dynamic-range cap: clamp lo upward so we display at most log_dr decades
        if log_dr is not None and log_dr > 0:
            lo_l = max(lo_l, hi_l - float(log_dr))
        with np.errstate(invalid="ignore", divide="ignore"):
            shifted = a + shift if shift else a
            bad = ~(shifted > 0)
            a_log = np.log10(np.where(bad, np.nan, shifted))
        denom = hi_l - lo_l
        if denom < _EPS:
            fill = 0.0 if hi <= 0 else 1.0
            norm = np.full_like(a, fill, dtype=np.float32)
        else:
            norm = (a_log - lo_l) * (1.0 / denom)
        nan_mask = nan_mask | ~np.isfinite(norm)
    else:
        denom = hi - lo
        if denom < _EPS:
            fill = 0.0 if hi <= 0 else 1.0
            norm = np.full_like(a, fill, dtype=np.float32)
        else:
            norm = (a - lo) * (1.0 / denom)

    np.clip(norm, 0.0, 1.0, out=norm)
    if nan_mask.any():
        norm[nan_mask] = 0.0
    return norm, nan_mask


def _gray_to_rgb_image(norm: np.ndarray, nan_mask: np.ndarray) -> Image.Image:
    g8 = (norm * 255).astype(np.uint8)
    if nan_mask.any():
        rgb = np.stack([g8, g8, g8], axis=-1)
        rgb[nan_mask] = (255, 0, 0)
        return Image.fromarray(rgb, mode="RGB")
    return Image.fromarray(g8, mode="L").convert("RGB")


def _tinted_gray_image(
    norm: np.ndarray, nan_mask: np.ndarray, wavelength: float
) -> Image.Image:
    rgb = tint_gray_by_wavelength((norm * 255).astype(np.uint8), wavelength)
    if nan_mask.any():
        rgb[nan_mask] = (255, 0, 0)
    return Image.fromarray(rgb, mode="RGB")


def _mask_image(arr: np.ndarray) -> Image.Image:
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


def _render_rgb_image(arr: np.ndarray) -> Image.Image:
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


def _render_multi_image(
    arr: np.ndarray,
    bands: tuple[int, ...] | None,
    wavelengths: list[float] | None,
    scale: str,
    clip_pct: tuple[float, float],
    log_dr: float | None = None,
) -> tuple[Image.Image, list[dict[str, Any]], str]:
    """Returns (composite_image, per_band_records, note)."""
    n_ch = arr.shape[2]

    # Per-band normalization for the strip + (if no wavelengths) the composite
    per_band: list[dict[str, Any]] = []
    norms: list[np.ndarray] = []
    for i in range(n_ch):
        band = arr[..., i]
        lo, hi = _compute_clip(band, clip_pct)
        norm, nan_mask = _normalize_with_clip(band, scale, lo, hi, log_dr=log_dr)
        norms.append(norm)
        # Use wavelength tint if available, else plain gray
        wl = wavelengths[i] if (wavelengths and i < len(wavelengths)) else None
        if wl is not None:
            img = _tinted_gray_image(norm, nan_mask, wl)
            band_label = f"ch {i} ({wl:g}nm)"
        else:
            img = _gray_to_rgb_image(norm, nan_mask)
            band_label = f"ch {i}"
        per_band.append({
            "label": band_label,
            "image": img,
            "stats": stats(band),
            "clip": [float(lo), float(hi)],
            "wavelength": wl,
        })

    # Composite
    if wavelengths and len(wavelengths) == n_ch:
        composite = composite_multi_by_wavelengths(norms, list(wavelengths))
        composite_img = Image.fromarray(composite, mode="RGB")
        note = f"{n_ch} channels, wavelength-weighted composite"
    else:
        # Classic R/G/B picks
        sel = tuple(bands) if bands else tuple(range(min(3, n_ch)))
        sel = tuple(max(0, min(n_ch - 1, int(b))) for b in sel)
        while len(sel) < 3:
            sel = (*sel, sel[-1])
        chs = [(norms[sel[i]] * 255).astype(np.uint8) for i in range(3)]
        composite_img = Image.fromarray(np.stack(chs, axis=-1), mode="RGB")
        note = f"{n_ch} channels, composite R:ch{sel[0]} G:ch{sel[1]} B:ch{sel[2]}"

    return composite_img, per_band, note


def render_array(
    x: Any,
    *,
    kind: str = "auto",
    scale: str = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
    log_dr: float | None = None,
    bands: tuple[int, ...] | None = None,
    wavelength: float | None = None,
    wavelengths: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Render an array into a PIL Image + metadata.

    Returns dict with: image (PIL), kind, shape, dtype, stats, scale,
    clip (the percentile-resolved [lo, hi] actually used), lossless (bool),
    note, bands (list for multi)."""
    if kind not in KINDS:
        kind = "auto"
    if isinstance(wavelengths, (tuple, list)):
        wavelengths = [float(w) for w in wavelengths]

    arr = np.asarray(x)
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    s = stats(arr)
    resolved = _resolve_kind(arr, kind)
    note: str | None = None
    lossless = False
    clip_used = (0.0, 1.0)
    bands_data: list[dict[str, Any]] | None = None

    if resolved == "mask":
        img = _mask_image(arr)
        lossless = True

    elif resolved in ("gray", "mono", "raw"):
        if arr.ndim == 3 and bands and len(bands) == 1:
            # Single-band slice of a 3D cube (band-strip click)
            arr = arr[..., int(bands[0])]
            shape = list(arr.shape)
        if arr.ndim != 2:
            raise ValueError(f"kind={resolved!r} expects 2-D, got ndim={arr.ndim}")
        lo, hi = _compute_clip(arr, clip_pct)
        norm, nan_mask = _normalize_with_clip(arr, scale, lo, hi, log_dr=log_dr)
        if wavelength is not None:
            img = _tinted_gray_image(norm, nan_mask, float(wavelength))
            note = f"tinted at {float(wavelength):g} nm"
        else:
            img = _gray_to_rgb_image(norm, nan_mask)
        clip_used = (lo, hi)
        if np.issubdtype(arr.dtype, np.floating) and nan_mask.any():
            lossless = True
        if resolved == "raw":
            note = (note + "; " if note else "") + "raw (no demosaic)"

    elif resolved == "rgb":
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"kind='rgb' expects (H,W,>=3), got {arr.shape}")
        img = _render_rgb_image(arr[..., :3])

    elif resolved == "multi":
        if arr.ndim != 3:
            raise ValueError(f"kind='multi' expects 3-D, got ndim={arr.ndim}")
        img, per_band, note = _render_multi_image(arr, bands, wavelengths, scale, clip_pct, log_dr)
        if not np.isfinite(arr).all():
            lossless = True
        bands_data = per_band

    else:
        raise ValueError(f"unknown kind {resolved!r}")

    return {
        "image": img,
        "kind": resolved,
        "shape": shape,
        "dtype": dtype,
        "stats": s,
        "scale": scale,
        "clip": [float(clip_used[0]), float(clip_used[1])],
        "clip_pct": [float(clip_pct[0]), float(clip_pct[1])],
        "log_dr": None if log_dr is None else float(log_dr),
        "lossless": lossless,
        "note": note,
        "bands": bands_data,
        "wavelength": wavelength,
        "wavelengths": wavelengths,
    }
