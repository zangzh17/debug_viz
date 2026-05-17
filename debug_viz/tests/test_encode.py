import io

import numpy as np
from PIL import Image

from debug_viz.encode import encode_array, THUMB_MAX


def _open(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def test_gray_2d(gray_2d):
    out = encode_array(gray_2d)
    assert out["kind"] == "gray"
    assert out["fmt"] == "jpeg"
    img = _open(out["main_full"])
    assert img.size == (60, 40)
    assert out["shape"] == [40, 60]
    assert out["dtype"].startswith("float")
    assert "min" in out["stats"]
    thumb = _open(out["main_thumb"])
    assert max(thumb.size) <= THUMB_MAX


def test_gray_nan_overlay(gray_with_nan):
    out = encode_array(gray_with_nan)
    img = np.array(_open(out["main_full"]).convert("RGB"))
    assert tuple(img[0, 0]) == (255, 0, 0)
    assert out["stats"]["nan_pct"] > 0


def test_log_scale(gray_2d):
    out = encode_array(gray_2d, scale="log")
    assert out["scale"] == "log"
    assert _open(out["main_full"]).size == (60, 40)


def test_rgb_3d(rgb_3d):
    out = encode_array(rgb_3d)
    assert out["kind"] == "rgb"
    img = _open(out["main_full"]).convert("RGB")
    assert img.size == (30, 20)
    assert out["shape"] == [20, 30, 3]


def test_mask_auto(mask_2d):
    out = encode_array(mask_2d)
    assert out["kind"] == "mask"
    img = np.array(_open(out["main_full"]).convert("RGB"))
    assert tuple(img[7, 7]) == (255, 220, 0)
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_multi_auto():
    rng = np.random.default_rng(0)
    arr = rng.random((30, 40, 5), dtype=np.float32)
    out = encode_array(arr)
    assert out["kind"] == "multi"
    assert "bands" in out and len(out["bands"]) == 5
    for b in out["bands"]:
        # JPEG magic
        assert b["full"][:3] == b"\xff\xd8\xff"
        bimg = _open(b["full"])
        assert bimg.size == (40, 30)


def test_multi_with_bands_selection():
    arr = np.random.default_rng(0).random((20, 25, 7), dtype=np.float32)
    out = encode_array(arr, bands=(0, 3, 6))
    assert out["kind"] == "multi"
    assert "ch0" in (out["note"] or "")
    assert "ch3" in (out["note"] or "")
    assert "ch6" in (out["note"] or "")
    assert len(out["bands"]) == 7


def test_explicit_kind_mono(gray_2d):
    out = encode_array(gray_2d, kind="mono")
    assert out["kind"] == "mono"
    assert _open(out["main_full"]).size == (60, 40)


def test_explicit_kind_raw(gray_2d):
    out = encode_array(gray_2d, kind="raw")
    assert out["kind"] == "raw"
    assert "demosaic" in (out["note"] or "")


def test_explicit_kind_gray_on_3d_fails_gracefully():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    out = encode_array(arr, kind="gray")
    assert "encode error" in (out["note"] or "") or out["kind"] == "gray"


def test_thumb_aspect_preserved():
    arr = np.zeros((4000, 1000), dtype=np.float32)
    out = encode_array(arr)
    thumb = _open(out["main_thumb"])
    assert max(thumb.size) <= THUMB_MAX
    assert abs((thumb.size[1] / thumb.size[0]) - 4.0) < 0.1


def test_stats_full_resolution(big_2d):
    out = encode_array(big_2d)
    assert out["stats"]["min"] == float(big_2d.min())
    assert out["stats"]["max"] == float(big_2d.max())


def test_constant_array():
    arr = np.ones((20, 20), dtype=np.float32) * 0.5
    out = encode_array(arr)
    img = np.array(_open(out["main_full"]).convert("RGB"))
    # Constant -> norm=1 -> bright. JPEG q=95 ~ exact 255.
    assert (img > 240).all()


def test_png_format_supported():
    arr = np.zeros((20, 20), dtype=np.float32)
    out = encode_array(arr, fmt="png")
    assert out["fmt"] == "png"
    assert out["main_full"][:8] == b"\x89PNG\r\n\x1a\n"


def test_downsample_path(big_2d):
    """Thumbnail produced via numpy downsampling, not double PIL encode."""
    out = encode_array(big_2d)
    thumb_img = _open(out["main_thumb"])
    # full is 1200x1500, max_dim 400 -> stride 3 -> ~400x500
    assert max(thumb_img.size) <= THUMB_MAX
    assert thumb_img.size[0] > 100 and thumb_img.size[1] > 100
