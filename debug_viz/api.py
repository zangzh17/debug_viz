"""Public API: view / compare / save.

Sends the raw numpy array (.npy bytes) + parameters to the server, which does
all rendering + DZI pyramid generation. This keeps the debugger-paused process
unblocked sooner (transfer over localhost is ~1-2 GB/s)."""
from __future__ import annotations

import io
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

from .boot import base_url, ensure_server

_POST_TIMEOUT = 60.0
_FALLBACK_DIR = Path("/tmp/debug_viz")


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _fallback_save(arr: np.ndarray, label: str | None) -> Path:
    """Server unreachable — best-effort render to a PNG on disk."""
    from .encode import render_array
    _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{(label or 'frame').replace('/', '_')}.png"
    p = _FALLBACK_DIR / name
    try:
        result = render_array(arr)
        result["image"].save(p, format="PNG")
    except Exception:
        # last-resort: dump raw shape info
        p.write_text(f"unsupported: shape={getattr(arr, 'shape', None)}")
    return p


def _panel_payload(
    arr: np.ndarray,
    *,
    label: str | None,
    kind: str,
    scale: str,
    clip_pct: tuple[float, float],
    log_dr: float | None,
    bands: tuple[int, ...] | None,
    wavelength: float | None,
    wavelengths: list[float] | tuple[float, ...] | None,
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "label": label or "",
        "kind": kind,
        "scale": scale,
        "clip_pct": list(clip_pct),
        "log_dr": log_dr,
        "bands": list(bands) if bands else None,
        "wavelength": wavelength,
        "wavelengths": list(wavelengths) if wavelengths else None,
    }


def view(
    x: Any,
    *,
    label: str | None = None,
    kind: Literal["auto", "gray", "mono", "rgb", "raw", "multi"] = "auto",
    bands: tuple[int, ...] | None = None,
    wavelength: float | None = None,
    wavelengths: list[float] | tuple[float, ...] | None = None,
    scale: Literal["linear", "log"] = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
    log_dr: float | None = None,
) -> None:
    try:
        arr = np.asarray(x)
    except Exception as e:
        print(f"[debug_viz] cannot convert to array: {e}")
        return

    panel = _panel_payload(
        arr, label=label, kind=kind, scale=scale, clip_pct=clip_pct,
        log_dr=log_dr, bands=bands, wavelength=wavelength, wavelengths=wavelengths,
    )
    meta = _sanitize_for_json({
        "id": uuid.uuid4().hex,
        "label": label or "",
        "ts": time.time(),
        "mode": "single",
        "panels": [panel],
    })

    if not ensure_server(timeout_sec=5.0):
        p = _fallback_save(arr, label)
        print(f"[debug_viz] server unavailable; saved {p}")
        return

    try:
        files = {
            "metadata": (None, json.dumps(meta), "application/json"),
            f"raw_{panel['id']}": (f"{panel['id']}.npy", _npy_bytes(arr),
                                   "application/octet-stream"),
        }
        httpx.post(f"{base_url()}/event", files=files, timeout=_POST_TIMEOUT)
    except Exception as e:
        p = _fallback_save(arr, label)
        print(f"[debug_viz] post failed ({e}); saved {p}")


def compare(
    *arrays: Any,
    label: str | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
    kind: Literal["auto", "gray", "mono", "rgb", "raw", "multi"] = "auto",
    bands: tuple[int, ...] | None = None,
    wavelength: float | None = None,
    wavelengths: list[float] | tuple[float, ...] | None = None,
    scale: Literal["linear", "log"] = "linear",
    clip_pct: tuple[float, float] = (1.0, 99.0),
    log_dr: float | None = None,
) -> None:
    """Compare 2-3 arrays side by side. Each gets its own panel + sub-title.

    compare(a, b)                        # 2-way
    compare(a, b, c, labels=("input", "denoised", "gt"))   # 3-way
    compare(a, b, label="before / after")                  # backward-compat
    """
    if len(arrays) < 2:
        print("[debug_viz] compare requires at least 2 arrays")
        return
    if len(arrays) > 4:
        print(f"[debug_viz] compare supports up to 4 arrays (got {len(arrays)})")
        return

    try:
        nps = [np.asarray(x) for x in arrays]
    except Exception as e:
        print(f"[debug_viz] cannot convert to arrays: {e}")
        return

    sublabels = list(labels) if labels else [f"{chr(ord('A') + i)}" for i in range(len(nps))]
    panels = [
        _panel_payload(
            a, label=sublabels[i] if i < len(sublabels) else f"{i}",
            kind=kind, scale=scale, clip_pct=clip_pct, log_dr=log_dr,
            bands=bands, wavelength=wavelength, wavelengths=wavelengths,
        )
        for i, a in enumerate(nps)
    ]
    meta = _sanitize_for_json({
        "id": uuid.uuid4().hex,
        "label": label or "",
        "ts": time.time(),
        "mode": "compare",
        "panels": panels,
    })

    if not ensure_server(timeout_sec=5.0):
        for a, p in zip(nps, panels):
            fp = _fallback_save(a, f"{label or 'cmp'}_{p['label']}")
            print(f"[debug_viz] server unavailable; saved {fp}")
        return

    try:
        files = {"metadata": (None, json.dumps(meta), "application/json")}
        for a, p in zip(nps, panels):
            files[f"raw_{p['id']}"] = (f"{p['id']}.npy", _npy_bytes(a),
                                       "application/octet-stream")
        httpx.post(f"{base_url()}/event", files=files, timeout=_POST_TIMEOUT)
    except Exception as e:
        print(f"[debug_viz] post failed: {e}")


def save(path: str | Path) -> None:
    p = Path(os.path.expanduser(str(path)))
    if not ensure_server(timeout_sec=2.0):
        print("[debug_viz] server not running; nothing to save")
        return
    try:
        httpx.post(f"{base_url()}/snapshot", json={"path": str(p)}, timeout=120.0)
        print(f"[debug_viz] saved snapshot -> {p}")
    except Exception as e:
        print(f"[debug_viz] save failed: {e}")
