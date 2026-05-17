"""Wavelength → RGB helper tests."""
import numpy as np

from debug_viz.spectral import (
    composite_multi_by_wavelengths,
    tint_gray_by_wavelength,
    wavelength_to_rgb,
)


def test_visible_red():
    r, g, b = wavelength_to_rgb(660)
    assert r > 0.9 and g < 0.1 and b < 0.1


def test_visible_green():
    r, g, b = wavelength_to_rgb(530)
    assert g > 0.9 and r < 0.8 and b < 0.2


def test_visible_blue():
    r, g, b = wavelength_to_rgb(460)
    assert b > 0.9 and r < 0.2


def test_uv_attenuated():
    r, g, b = wavelength_to_rgb(280)
    # UV: deep violet, attenuated
    assert b > 0 and b < 1.0
    assert r > 0


def test_ir_attenuated():
    r, g, b = wavelength_to_rgb(900)
    # IR: dark red
    assert g == 0 and b == 0
    assert 0 < r < 1.0


def test_edge_falloff():
    # Near edges of visible band, brightness attenuates
    edge_r, _, _ = wavelength_to_rgb(395)  # near UV edge
    mid_r, _, _ = wavelength_to_rgb(660)
    assert mid_r > edge_r * 1.5  # mid-band visibly brighter


def test_tint_gray():
    g = np.full((10, 10), 0.8, dtype=np.float32)
    rgb = tint_gray_by_wavelength(g, 660)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
    # Red dominant
    assert rgb[..., 0].mean() > 150
    assert rgb[..., 1].mean() < 30
    assert rgb[..., 2].mean() < 30


def test_composite_multi():
    # Two bands: red and blue, both bright in different regions
    red = np.zeros((20, 20), dtype=np.float32); red[:, :10] = 1.0
    blue = np.zeros((20, 20), dtype=np.float32); blue[:, 10:] = 1.0
    out = composite_multi_by_wavelengths([red, blue], [660, 460])
    assert out.shape == (20, 20, 3)
    # Left half: dominantly red
    assert out[10, 5, 0] > out[10, 5, 2]
    # Right half: dominantly blue
    assert out[10, 15, 2] > out[10, 15, 0]


def test_composite_normalizes_peak():
    a = np.ones((4, 4), dtype=np.float32)
    out = composite_multi_by_wavelengths([a, a], [550, 550])
    # Peak should be ~255 regardless of how many bands accumulate
    assert out.max() <= 255 and out.max() > 200
