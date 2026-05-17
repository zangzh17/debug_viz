import io

import numpy as np
from PIL import Image

from debug_viz.encode import encode_array, THUMB_MAX


def _open(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def test_gray_2d(gray_2d):
    out = encode_array(gray_2d)
    img = _open(out["full_png"])
    assert img.size == (60, 40)
    assert out["shape"] == [40, 60]
    assert out["dtype"].startswith("float")
    assert "min" in out["stats"]
    assert out["scale"] == "linear"
    thumb = _open(out["thumb_png"])
    assert max(thumb.size) <= THUMB_MAX


def test_gray_nan_overlay(gray_with_nan):
    out = encode_array(gray_with_nan)
    img = np.array(_open(out["full_png"]))
    # NaN pixel was set at (0,0) and (5,5) -> red
    assert img[0, 0, 0] == 255 and img[0, 0, 1] == 0 and img[0, 0, 2] == 0
    assert out["stats"]["nan_pct"] > 0


def test_log_scale(gray_2d):
    out = encode_array(gray_2d, scale="log")
    assert out["scale"] == "log"
    assert _open(out["full_png"]).size == (60, 40)


def test_rgb_3d(rgb_3d):
    out = encode_array(rgb_3d)
    img = _open(out["full_png"])
    assert img.mode == "RGB"
    assert img.size == (30, 20)
    assert out["shape"] == [20, 30, 3]


def test_mask_2d(mask_2d):
    out = encode_array(mask_2d)
    img = np.array(_open(out["full_png"]))
    # truthy -> yellow (255, 220, 0)
    assert tuple(img[7, 7]) == (255, 220, 0)
    # falsy -> black
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_high_dim_first3():
    arr = np.random.default_rng(0).integers(0, 256, (10, 10, 5), dtype=np.uint8)
    out = encode_array(arr)
    assert "first 3 of 5" in (out["note"] or "")


def test_degrade_unsupported():
    out = encode_array(np.zeros((2, 2, 2, 2)))  # 4-D not handled
    assert out["note"] and "unsupported" in out["note"]


def test_thumb_aspect_preserved():
    arr = np.zeros((4000, 1000), dtype=np.float32)
    out = encode_array(arr)
    thumb = _open(out["thumb_png"])
    # max dim 800; preserves aspect
    assert max(thumb.size) <= THUMB_MAX
    ratio_in = 4000 / 1000
    ratio_out = thumb.size[1] / thumb.size[0]  # H/W
    assert abs(ratio_in - ratio_out) < 0.05


def test_stats_full_resolution(big_2d):
    out = encode_array(big_2d)
    assert out["stats"]["min"] == float(big_2d.min())
    assert out["stats"]["max"] == float(big_2d.max())


def test_constant_array():
    arr = np.ones((20, 20), dtype=np.float32) * 0.5
    out = encode_array(arr)
    img = np.array(_open(out["full_png"]))
    # constant -> norm=1 -> all 255
    assert (img == 255).all()
