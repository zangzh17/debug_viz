"""DZI (Deep Zoom Image) pyramid generator for OpenSeadragon.

Writes a multi-level tile pyramid for one PIL image:

    <out_dir>/<name>.dzi              # XML descriptor
    <out_dir>/<name>_files/<L>/<x>_<y>.<ext>

Tile size 256 (DZI default), overlap 0 (slight seams under extreme zoom,
but ~2x faster to write than overlap=1). Levels go from L=0 (1x1) up to
L=max_level (full resolution). OpenSeadragon requests them by level number.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

TILE_SIZE = 256
OVERLAP = 0
JPEG_QUALITY = 88


@dataclass
class DZIManifest:
    name: str
    width: int
    height: int
    tile_size: int
    overlap: int
    fmt: str
    max_level: int


def _xml_descriptor(width: int, height: int, fmt: str) -> str:
    ext = "jpg" if fmt == "jpeg" else fmt
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        f'  TileSize="{TILE_SIZE}" Overlap="{OVERLAP}" Format="{ext}">\n'
        f'  <Size Width="{width}" Height="{height}"/>\n'
        '</Image>\n'
    )


def _save_tile(img: Image.Image, path: Path, fmt: str) -> None:
    if fmt == "jpeg":
        img.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=False)
    elif fmt == "png":
        img.save(path, format="PNG", compress_level=1)
    else:
        raise ValueError(f"unsupported tile fmt: {fmt}")


def write_dzi(
    img: Image.Image,
    out_dir: Path,
    name: str,
    fmt: str = "jpeg",
) -> DZIManifest:
    """Write a complete DZI pyramid for img. Returns the manifest.

    Reuses the previous level's downsampled image to avoid re-downsampling
    from the original every time (big win for huge images)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    max_dim = max(w, h)
    # DZI level L has size ceil(orig / 2^(max_level - L))
    max_level = int(math.ceil(math.log2(max_dim))) if max_dim > 1 else 0
    files_dir = out_dir / f"{name}_files"
    files_dir.mkdir(exist_ok=True)
    ext = "jpg" if fmt == "jpeg" else fmt

    # Build levels from top (full res) downward; reuse previous level image
    current = img
    for level in range(max_level, -1, -1):
        scale = 2 ** (max_level - level)
        lvl_w = max(1, math.ceil(w / scale))
        lvl_h = max(1, math.ceil(h / scale))
        if current.size != (lvl_w, lvl_h):
            current = current.resize((lvl_w, lvl_h), Image.LANCZOS if scale <= 4 else Image.BILINEAR)
        lvl_dir = files_dir / str(level)
        lvl_dir.mkdir(exist_ok=True)

        n_cols = math.ceil(lvl_w / TILE_SIZE)
        n_rows = math.ceil(lvl_h / TILE_SIZE)
        for ty in range(n_rows):
            for tx in range(n_cols):
                x0 = tx * TILE_SIZE
                y0 = ty * TILE_SIZE
                x1 = min(x0 + TILE_SIZE, lvl_w)
                y1 = min(y0 + TILE_SIZE, lvl_h)
                tile = current.crop((x0, y0, x1, y1))
                _save_tile(tile, lvl_dir / f"{tx}_{ty}.{ext}", fmt)

    (out_dir / f"{name}.dzi").write_text(_xml_descriptor(w, h, fmt))
    return DZIManifest(
        name=name, width=w, height=h,
        tile_size=TILE_SIZE, overlap=OVERLAP, fmt=fmt, max_level=max_level,
    )


def thumbnail_bytes(img: Image.Image, max_dim: int = 400, fmt: str = "jpeg") -> bytes:
    """Cheap thumbnail for gallery tile."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    s = max(1, math.ceil(max(w, h) / max_dim))
    thumb = img.resize((max(1, w // s), max(1, h // s)), Image.BILINEAR) if s > 1 else img
    buf = io.BytesIO()
    if fmt == "jpeg":
        thumb.save(buf, format="JPEG", quality=85)
    else:
        thumb.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
