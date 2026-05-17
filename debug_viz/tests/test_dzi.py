"""DZI pyramid generator tests."""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from debug_viz.dzi import TILE_SIZE, thumbnail_bytes, write_dzi


def _make_image(w, h, mode="RGB"):
    arr = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def test_descriptor_exists(tmp_path):
    img = _make_image(512, 256)
    m = write_dzi(img, tmp_path, "frame")
    desc = tmp_path / "frame.dzi"
    assert desc.exists()
    root = ET.fromstring(desc.read_text())
    size = root.find("{http://schemas.microsoft.com/deepzoom/2008}Size")
    assert int(size.get("Width")) == 512
    assert int(size.get("Height")) == 256


def test_max_level_matches_image(tmp_path):
    img = _make_image(2000, 1500)
    m = write_dzi(img, tmp_path, "frame")
    assert m.max_level == int(math.ceil(math.log2(2000)))


def test_tile_count_bottom_level(tmp_path):
    img = _make_image(513, 257)  # 3x2 tiles at bottom
    m = write_dzi(img, tmp_path, "frame")
    bottom = tmp_path / "frame_files" / str(m.max_level)
    tiles = list(bottom.glob("*.jpg"))
    # 513/256=ceil 3, 257/256=ceil 2 → 6 tiles
    assert len(tiles) == 6


def test_lossless_png_path(tmp_path):
    img = _make_image(128, 128)
    m = write_dzi(img, tmp_path, "frame", fmt="png")
    assert m.fmt == "png"
    bottom = tmp_path / "frame_files" / str(m.max_level)
    pngs = list(bottom.glob("*.png"))
    assert pngs and pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_each_level_halved(tmp_path):
    img = _make_image(1024, 1024)
    m = write_dzi(img, tmp_path, "frame")
    # Top level (smallest): 1x1 tile, very small image
    top_tile = tmp_path / "frame_files" / "0" / "0_0.jpg"
    assert top_tile.exists()
    pil = Image.open(top_tile)
    assert max(pil.size) <= TILE_SIZE


def test_thumbnail_bytes_jpeg():
    img = _make_image(2000, 1000)
    b = thumbnail_bytes(img, max_dim=400)
    assert b[:3] == b"\xff\xd8\xff"  # JPEG magic
    pil = Image.open(__import__("io").BytesIO(b))
    assert max(pil.size) <= 400


def test_thumbnail_aspect_preserved():
    img = _make_image(800, 200)
    b = thumbnail_bytes(img, max_dim=400)
    pil = Image.open(__import__("io").BytesIO(b))
    assert pil.size == (400, 100)
