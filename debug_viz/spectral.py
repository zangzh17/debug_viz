"""Wavelength -> sRGB approximation + per-band composite for spectral data.

Visible range 380-780 nm uses the Bruton/Dan Bruton approximation (good enough
for scientific viz tinting — not photometrically accurate). UV (<380) clamps
to deep violet, IR (>780) to deep red.
"""
from __future__ import annotations

import numpy as np

VIS_MIN = 380.0
VIS_MAX = 780.0


def wavelength_to_rgb(nm: float) -> tuple[float, float, float]:
    """Map a wavelength in nm to an (r, g, b) tuple in [0, 1].

    Out-of-visible-band wavelengths get clamped to nearest visible edge color,
    then attenuated (UV/IR are not directly perceivable)."""
    w = float(nm)

    if w < VIS_MIN:
        # UV: deep violet, attenuated as we go deeper into UV
        att = max(0.2, 1.0 - (VIS_MIN - w) / 200.0)
        return (0.5 * att, 0.0, 1.0 * att)
    if w > VIS_MAX:
        # IR: dark red, attenuated
        att = max(0.15, 1.0 - (w - VIS_MAX) / 400.0)
        return (1.0 * att, 0.0, 0.0)

    if w < 440:
        r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif w < 580:
        r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    else:  # 645..780
        r, g, b = 1.0, 0.0, 0.0

    # Edge falloff (mimic CIE photopic response near limits)
    if w < 420:
        att = 0.3 + 0.7 * (w - 380) / 40.0
    elif w > 700:
        att = 0.3 + 0.7 * (780 - w) / 80.0
    else:
        att = 1.0
    return (r * att, g * att, b * att)


def tint_gray_by_wavelength(gray: np.ndarray, nm: float) -> np.ndarray:
    """Apply a wavelength tint to a [0,1] (or uint8) grayscale image.

    Returns uint8 (H, W, 3)."""
    r, g, b = wavelength_to_rgb(nm)
    if gray.dtype == np.uint8:
        g8 = gray
    else:
        g8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    out = np.empty((*g8.shape, 3), dtype=np.uint8)
    # Use float32 for the multiply, then back to uint8
    f = g8.astype(np.float32) * (1.0 / 255.0)
    out[..., 0] = (f * r * 255).astype(np.uint8)
    out[..., 1] = (f * g * 255).astype(np.uint8)
    out[..., 2] = (f * b * 255).astype(np.uint8)
    return out


def composite_multi_by_wavelengths(
    norm_bands: list[np.ndarray],
    wavelengths: list[float],
) -> np.ndarray:
    """Sum-composite N normalized [0,1] bands tinted by their wavelengths.

    Each band's per-pixel intensity is scaled by its wavelength's sRGB color and
    accumulated. The result is normalized so the brightest pixel stays at 255.

    norm_bands: list of (H, W) float32 in [0, 1]
    wavelengths: list of nm (same length as norm_bands)

    Returns uint8 (H, W, 3).
    """
    assert len(norm_bands) == len(wavelengths)
    if not norm_bands:
        raise ValueError("need at least one band")
    h, w = norm_bands[0].shape
    acc = np.zeros((h, w, 3), dtype=np.float32)
    for band, nm in zip(norm_bands, wavelengths):
        r, g, b = wavelength_to_rgb(nm)
        acc[..., 0] += band * r
        acc[..., 1] += band * g
        acc[..., 2] += band * b
    peak = float(acc.max())
    if peak <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    # Normalize the brightest channel pixel to 255 (preserve hue)
    acc *= (255.0 / peak)
    np.clip(acc, 0, 255, out=acc)
    return acc.astype(np.uint8)
