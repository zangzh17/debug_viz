"""Tests for the pure-render core (encode.render_array)."""
import numpy as np
from PIL import Image

from debug_viz.encode import render_array


def test_gray_2d(gray_2d):
    out = render_array(gray_2d)
    assert out["kind"] == "gray"
    assert isinstance(out["image"], Image.Image)
    assert out["image"].size == (60, 40)
    assert out["shape"] == [40, 60]
    assert out["dtype"].startswith("float")
    assert "min" in out["stats"]
    assert out["lossless"] is False
    assert out["scale"] == "linear"


def test_gray_nan_overlay(gray_with_nan):
    out = render_array(gray_with_nan)
    img = np.array(out["image"].convert("RGB"))
    assert tuple(img[0, 0]) == (255, 0, 0)
    assert out["stats"]["nan_pct"] > 0
    assert out["lossless"] is True  # NaN forces PNG-quality output


def test_log_scale(gray_2d):
    out = render_array(gray_2d, scale="log")
    assert out["scale"] == "log"
    assert out["image"].size == (60, 40)


def test_clip_pct_applied(big_2d):
    out = render_array(big_2d, clip_pct=(5.0, 95.0))
    lo, hi = out["clip"]
    # 5/95 percentile band must be narrower than 0/100
    assert lo > float(big_2d.min())
    assert hi < float(big_2d.max())
    assert out["clip_pct"] == [5.0, 95.0]


def test_rgb_3d(rgb_3d):
    out = render_array(rgb_3d)
    assert out["kind"] == "rgb"
    assert out["image"].size == (30, 20)
    assert out["shape"] == [20, 30, 3]


def test_mask_auto(mask_2d):
    out = render_array(mask_2d)
    assert out["kind"] == "mask"
    img = np.array(out["image"].convert("RGB"))
    assert tuple(img[7, 7]) == (255, 220, 0)
    assert tuple(img[0, 0]) == (0, 0, 0)
    assert out["lossless"] is True


def test_multi_auto():
    arr = np.random.default_rng(0).random((30, 40, 5), dtype=np.float32)
    out = render_array(arr)
    assert out["kind"] == "multi"
    assert out["bands"] is not None and len(out["bands"]) == 5
    for b in out["bands"]:
        assert isinstance(b["image"], Image.Image)
        assert b["image"].size == (40, 30)
        assert b["wavelength"] is None


def test_multi_with_bands_selection():
    arr = np.random.default_rng(0).random((20, 25, 7), dtype=np.float32)
    out = render_array(arr, bands=(0, 3, 6))
    assert out["kind"] == "multi"
    note = out["note"] or ""
    assert "ch0" in note and "ch3" in note and "ch6" in note


def test_multi_with_wavelengths():
    """All-bands wavelength composite → uses wavelength tinting, not R/G/B picks."""
    arr = np.zeros((40, 50, 3), dtype=np.float32)
    arr[..., 0] = 1.0  # band 0 = full intensity
    out = render_array(arr, kind="multi", wavelengths=[450, 550, 650])
    assert out["bands"][0]["wavelength"] == 450
    assert "wavelength-weighted" in (out["note"] or "")
    # band 0 (450nm = blue) lit → composite should be blue-dominant
    pix = np.array(out["image"])[20, 25]
    assert pix[2] > pix[0]  # blue > red


def test_mono_tinted_by_wavelength(gray_2d):
    out = render_array(gray_2d, kind="mono", wavelength=656)
    assert out["kind"] == "mono"
    assert "656" in (out["note"] or "")
    img = np.array(out["image"])
    # 656nm = pure red, so green/blue channels should be ~0
    bright = img.reshape(-1, 3).max(axis=0)
    assert bright[0] > 100   # red present
    assert bright[1] < 30    # green suppressed
    assert bright[2] < 30    # blue suppressed


def test_explicit_kind_raw(gray_2d):
    out = render_array(gray_2d, kind="raw")
    assert out["kind"] == "raw"
    assert "raw" in (out["note"] or "")


def test_constant_array():
    arr = np.ones((20, 20), dtype=np.float32) * 0.5
    out = render_array(arr)
    img = np.array(out["image"].convert("RGB"))
    assert (img >= 250).all()


def test_stats_includes_finite_only(gray_with_nan):
    out = render_array(gray_with_nan)
    assert np.isfinite(out["stats"]["min"])
    assert np.isfinite(out["stats"]["max"])


def test_kind_explicit_gray_raises_on_3d():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    try:
        render_array(arr, kind="gray")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
