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
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _fallback_write(data: bytes, label: str | None, ext: str = "webp") -> Path:
    _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{(label or 'frame').replace('/', '_')}.{ext}"
    p = _FALLBACK_DIR / name
    p.write_bytes(data)
    return p


def view(
    x: Any,
    *,
    label: str | None = None,
    kind: Literal["auto", "gray", "mono", "rgb", "raw", "multi"] = "auto",
    bands: tuple[int, ...] | None = None,
    scale: Literal["linear", "log"] = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
) -> None:
    try:
        enc = encode_array(x, kind=kind, bands=bands, scale=scale, clip_pct=clip_pct)
    except Exception as e:
        print(f"[debug_viz] encode failed: {e}")
        return

    band_meta = None
    if enc.get("bands"):
        band_meta = [{"label": b["label"], "stats": b["stats"], "clip": b["clip"]}
                     for b in enc["bands"]]

    meta = {
        "id": uuid.uuid4().hex,
        "label": label or "",
        "ts": time.time(),
        "mode": "single",
        "kind": enc["kind"],
        "fmt": enc["fmt"],
        "shape": enc["shape"],
        "dtype": enc["dtype"],
        "stats": enc["stats"],
        "scale": enc["scale"],
        "clip": enc["clip"],
        "note": enc["note"],
        "bands_selected": list(bands) if bands else None,
        "n_bands": len(enc["bands"]) if enc.get("bands") else 0,
        "band_meta": band_meta,
    }
    meta = _sanitize_for_json(meta)

    if not ensure_server(timeout_sec=5.0):
        p = _fallback_write(enc["main_full"], label, ext=enc["fmt"])
        print(f"[debug_viz] server unavailable; saved {p}")
        return

    try:
        files = {
            "metadata": (None, json.dumps(meta), "application/json"),
            "main_full": (f"main.{enc['fmt']}", enc["main_full"], f"image/{enc['fmt']}"),
            "main_thumb": (f"main_t.{enc['fmt']}", enc["main_thumb"], f"image/{enc['fmt']}"),
        }
        for i, b in enumerate(enc.get("bands") or []):
            files[f"band_{i}_full"] = (f"b{i}.{enc['fmt']}", b["full"], f"image/{enc['fmt']}")
            files[f"band_{i}_thumb"] = (f"b{i}_t.{enc['fmt']}", b["thumb"], f"image/{enc['fmt']}")
        httpx.post(f"{base_url()}/event", files=files, timeout=_POST_TIMEOUT)
    except Exception as e:
        p = _fallback_write(enc["main_full"], label, ext=enc["fmt"])
        print(f"[debug_viz] post failed ({e}); saved {p}")


def compare(
    a: Any,
    b: Any,
    *,
    label: str | None = None,
    kind: Literal["auto", "gray", "mono", "rgb", "raw"] = "auto",
    scale: Literal["linear", "log"] = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
) -> None:
    try:
        enc_a = encode_array(a, kind=kind, scale=scale, clip_pct=clip_pct)
        enc_b = encode_array(b, kind=kind, scale=scale, clip_pct=clip_pct)
    except Exception as e:
        print(f"[debug_viz] encode failed: {e}")
        return

    fmt = enc_a["fmt"]
    meta = {
        "id": uuid.uuid4().hex,
        "label": label or "",
        "ts": time.time(),
        "mode": "compare",
        "kind": enc_a["kind"],
        "fmt": fmt,
        "shape": [enc_a["shape"], enc_b["shape"]],
        "dtype": [enc_a["dtype"], enc_b["dtype"]],
        "stats": [enc_a["stats"], enc_b["stats"]],
        "scale": scale,
        "clip": [enc_a["clip"], enc_b["clip"]],
        "note": enc_a["note"] or enc_b["note"],
    }
    meta = _sanitize_for_json(meta)

    if not ensure_server(timeout_sec=5.0):
        pa = _fallback_write(enc_a["main_full"], (label or "") + "_a", ext=fmt)
        pb = _fallback_write(enc_b["main_full"], (label or "") + "_b", ext=fmt)
        print(f"[debug_viz] server unavailable; saved {pa}, {pb}")
        return

    try:
        files = {
            "metadata": (None, json.dumps(meta), "application/json"),
            "full_a": (f"a.{fmt}", enc_a["main_full"], f"image/{fmt}"),
            "thumb_a": (f"at.{fmt}", enc_a["main_thumb"], f"image/{fmt}"),
            "full_b": (f"b.{fmt}", enc_b["main_full"], f"image/{fmt}"),
            "thumb_b": (f"bt.{fmt}", enc_b["main_thumb"], f"image/{fmt}"),
        }
        httpx.post(f"{base_url()}/event", files=files, timeout=_POST_TIMEOUT)
    except Exception as e:
        pa = _fallback_write(enc_a["main_full"], (label or "") + "_a", ext=fmt)
        pb = _fallback_write(enc_b["main_full"], (label or "") + "_b", ext=fmt)
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
