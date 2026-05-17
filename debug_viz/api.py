"""Public API: view / compare / save."""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx

from .boot import base_url, ensure_server
from .encode import encode_array

_POST_TIMEOUT = 30.0
_FALLBACK_DIR = Path("/tmp/debug_viz")


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None (httpx strict JSON encoder rejects them)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _fallback_write(png: bytes, label: str | None) -> Path:
    _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{(label or 'frame').replace('/', '_')}.png"
    p = _FALLBACK_DIR / name
    p.write_bytes(png)
    return p


def view(
    x: Any,
    *,
    label: str | None = None,
    scale: Literal["linear", "log"] = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
) -> None:
    try:
        enc = encode_array(x, scale=scale, clip_pct=clip_pct)
    except Exception as e:
        print(f"[debug_viz] encode failed: {e}")
        return

    meta = {
        "id": uuid.uuid4().hex,
        "label": label or "",
        "ts": time.time(),
        "mode": "single",
        "shape": enc["shape"],
        "dtype": enc["dtype"],
        "stats": enc["stats"],
        "scale": enc["scale"],
        "clip": enc["clip"],
        "note": enc["note"],
    }
    meta = _sanitize_for_json(meta)

    if not ensure_server(timeout_sec=5.0):
        p = _fallback_write(enc["full_png"], label)
        print(f"[debug_viz] server unavailable; saved PNG to {p}")
        return

    try:
        files = {
            "metadata": (None, json.dumps(meta), "application/json"),
            "full": ("full.png", enc["full_png"], "image/png"),
            "thumb": ("thumb.png", enc["thumb_png"], "image/png"),
        }
        httpx.post(f"{base_url()}/event", files=files, timeout=_POST_TIMEOUT)
    except Exception as e:
        p = _fallback_write(enc["full_png"], label)
        print(f"[debug_viz] post failed ({e}); saved PNG to {p}")


def compare(
    a: Any,
    b: Any,
    *,
    label: str | None = None,
    scale: Literal["linear", "log"] = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
) -> None:
    try:
        enc_a = encode_array(a, scale=scale, clip_pct=clip_pct)
        enc_b = encode_array(b, scale=scale, clip_pct=clip_pct)
    except Exception as e:
        print(f"[debug_viz] encode failed: {e}")
        return

    meta = {
        "id": uuid.uuid4().hex,
        "label": label or "",
        "ts": time.time(),
        "mode": "compare",
        "shape": [enc_a["shape"], enc_b["shape"]],
        "dtype": [enc_a["dtype"], enc_b["dtype"]],
        "stats": [enc_a["stats"], enc_b["stats"]],
        "scale": scale,
        "clip": [enc_a["clip"], enc_b["clip"]],
        "note": enc_a["note"] or enc_b["note"],
    }
    meta = _sanitize_for_json(meta)

    if not ensure_server(timeout_sec=5.0):
        pa = _fallback_write(enc_a["full_png"], (label or "") + "_a")
        pb = _fallback_write(enc_b["full_png"], (label or "") + "_b")
        print(f"[debug_viz] server unavailable; saved {pa}, {pb}")
        return

    try:
        files = {
            "metadata": (None, json.dumps(meta), "application/json"),
            "full_a": ("a.png", enc_a["full_png"], "image/png"),
            "thumb_a": ("a_t.png", enc_a["thumb_png"], "image/png"),
            "full_b": ("b.png", enc_b["full_png"], "image/png"),
            "thumb_b": ("b_t.png", enc_b["thumb_png"], "image/png"),
        }
        httpx.post(f"{base_url()}/event", files=files, timeout=_POST_TIMEOUT)
    except Exception as e:
        pa = _fallback_write(enc_a["full_png"], (label or "") + "_a")
        pb = _fallback_write(enc_b["full_png"], (label or "") + "_b")
        print(f"[debug_viz] post failed ({e}); saved {pa}, {pb}")


def save(path: str | Path) -> None:
    p = Path(os.path.expanduser(str(path)))
    if not ensure_server(timeout_sec=2.0):
        print("[debug_viz] server not running; nothing to save")
        return
    try:
        httpx.post(f"{base_url()}/snapshot", json={"path": str(p)}, timeout=30.0)
        print(f"[debug_viz] saved snapshot -> {p}")
    except Exception as e:
        print(f"[debug_viz] save failed: {e}")
