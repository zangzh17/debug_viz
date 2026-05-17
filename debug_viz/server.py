"""Starlette server: raw-array storage + DZI tile pyramid serving + WS broadcast.

Storage layout (under png_dir):
    raw/<panel_id>.npy           # original array, kept for re-rendering
    thumb/<panel_id>.<ext>       # small gallery thumbnail
    thumb/<panel_id>__b<i>.<ext> # per-band thumbs (multi data)
    dzi/<panel_id>/<render_id>.dzi          # DZI XML descriptor
    dzi/<panel_id>/<render_id>_files/...    # tile pyramid

Frames reference panels by id; panels reference the *current* render_id.
Re-rendering produces a fresh render_id (so HTTP cache is bypassed) and the
old DZI directory is garbage collected.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .dzi import thumbnail_bytes, write_dzi
from .encode import render_array

WEB_DIR = Path(__file__).parent / "web"
SESSION_DIR_BASE = Path("/tmp/debug_viz")

# v3 = panels + raw archive + DZI tiles
MANIFEST_VERSION = 3

_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".png": "image/png", ".webp": "image/webp"}


def _media_for(suffix: str) -> str:
    return _MEDIA.get(suffix.lower(), "application/octet-stream")


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


class Archive:
    def __init__(self, root: Path):
        self.root = root
        self.raw_dir = root / "raw"
        self.thumb_dir = root / "thumb"
        self.dzi_dir = root / "dzi"
        for d in (self.raw_dir, self.thumb_dir, self.dzi_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict[str, Any]] = []
        # panel_id -> dict (current panel state, includes render_id, current dzi)
        self.panels: dict[str, dict[str, Any]] = {}

    # ---- raw storage --------------------------------------------------

    def store_raw(self, panel_id: str, arr: np.ndarray) -> Path:
        p = self.raw_dir / f"{panel_id}.npy"
        np.save(p, arr, allow_pickle=False)
        return p

    def load_raw(self, panel_id: str) -> np.ndarray | None:
        p = self.raw_dir / f"{panel_id}.npy"
        if not p.exists():
            return None
        return np.load(p, allow_pickle=False)

    # ---- thumb / dzi --------------------------------------------------

    def write_thumb(self, asset_id: str, fmt: str, data: bytes) -> Path:
        ext = "jpg" if fmt == "jpeg" else fmt
        p = self.thumb_dir / f"{asset_id}.{ext}"
        p.write_bytes(data)
        return p

    def thumb_path(self, asset_id: str) -> Path | None:
        matches = list(self.thumb_dir.glob(f"{asset_id}.*"))
        return matches[0] if matches else None

    def dzi_panel_dir(self, panel_id: str) -> Path:
        d = self.dzi_dir / panel_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_dzi_for(
        self, panel_id: str, render_id: str, image, fmt: str,
    ) -> dict[str, Any]:
        out_dir = self.dzi_panel_dir(panel_id)
        m = write_dzi(image, out_dir, render_id, fmt=fmt)
        return {
            "width": m.width, "height": m.height,
            "tile_size": m.tile_size, "overlap": m.overlap,
            "fmt": m.fmt, "max_level": m.max_level,
        }

    def gc_old_renders(self, panel_id: str, keep_render_id: str) -> None:
        d = self.dzi_panel_dir(panel_id)
        for entry in d.iterdir():
            if entry.is_file() and entry.suffix == ".dzi" and entry.stem != keep_render_id:
                entry.unlink(missing_ok=True)
            elif entry.is_dir() and entry.name.endswith("_files") \
                    and not entry.name.startswith(f"{keep_render_id}_"):
                shutil.rmtree(entry, ignore_errors=True)

    # ---- frames -------------------------------------------------------

    def add_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        if not frame.get("id"):
            frame["id"] = _short_id()
        self.frames.append(frame)
        for p in frame.get("panels", []):
            self.panels[p["id"]] = p
        return frame

    def get_frame(self, fid: str) -> dict[str, Any] | None:
        for f in self.frames:
            if f["id"] == fid:
                return f
        return None

    def get_panel(self, pid: str) -> dict[str, Any] | None:
        return self.panels.get(pid)

    def update_frame(self, fid: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        f = self.get_frame(fid)
        if not f:
            return None
        if "label" in fields:
            f["label"] = fields["label"]
        return f

    def update_panel(self, pid: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        p = self.panels.get(pid)
        if not p:
            return None
        for k in ("label",):
            if k in fields:
                p[k] = fields[k]
        return p

    def delete_frame(self, fid: str) -> bool:
        for i, f in enumerate(self.frames):
            if f["id"] == fid:
                for panel in f.get("panels", []):
                    pid = panel["id"]
                    (self.raw_dir / f"{pid}.npy").unlink(missing_ok=True)
                    for tp in self.thumb_dir.glob(f"{pid}*"):
                        tp.unlink(missing_ok=True)
                    pdir = self.dzi_panel_dir(pid)
                    shutil.rmtree(pdir, ignore_errors=True)
                    self.panels.pop(pid, None)
                del self.frames[i]
                return True
        return False

    def list_frames(self) -> list[dict[str, Any]]:
        return [dict(f) for f in self.frames]

    # ---- snapshot -----------------------------------------------------

    def snapshot(self, out_path: Path) -> None:
        manifest = {
            "version": MANIFEST_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frames": self.list_frames(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            # raw arrays
            for f in self.frames:
                for panel in f.get("panels", []):
                    pid = panel["id"]
                    rp = self.raw_dir / f"{pid}.npy"
                    if rp.exists():
                        z.write(rp, arcname=f"raw/{rp.name}")
                    for tp in self.thumb_dir.glob(f"{pid}*"):
                        z.write(tp, arcname=f"thumb/{tp.name}")
                    # Pack current DZI directory
                    pdir = self.dzi_dir / pid
                    if pdir.exists():
                        for entry in pdir.rglob("*"):
                            if entry.is_file():
                                rel = entry.relative_to(self.dzi_dir)
                                z.write(entry, arcname=f"dzi/{rel}")
            z.writestr("manifest.json", json.dumps(manifest, indent=2))


# -------------------- render helpers --------------------

def _render_and_store(
    archive: Archive, panel: dict[str, Any], arr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render the panel's current params, write DZI pyramid + thumb, mutate panel.

    Returns the panel."""
    if arr is None:
        arr = archive.load_raw(panel["id"])
    if arr is None:
        raise FileNotFoundError(f"no raw for {panel['id']}")

    result = render_array(
        arr,
        kind=panel.get("kind", "auto"),
        scale=panel.get("scale", "linear"),
        clip_pct=tuple(panel.get("clip_pct", (1.0, 99.0))),
        log_dr=panel.get("log_dr"),
        bands=tuple(panel["bands"]) if panel.get("bands") else None,
        wavelength=panel.get("wavelength"),
        wavelengths=panel.get("wavelengths"),
    )
    fmt = "png" if result["lossless"] else "jpeg"

    render_id = _short_id()
    dzi_meta = archive.write_dzi_for(panel["id"], render_id, result["image"], fmt)

    # Composite thumb (the gallery thumbnail)
    thumb = thumbnail_bytes(result["image"], max_dim=320, fmt=fmt)
    archive.write_thumb(panel["id"], fmt, thumb)

    # Per-band thumbs for multi
    band_meta = None
    if result["bands"]:
        band_meta = []
        for i, b in enumerate(result["bands"]):
            bid = f"{panel['id']}__b{i}"
            archive.write_thumb(bid, fmt, thumbnail_bytes(b["image"], max_dim=160, fmt=fmt))
            band_meta.append({
                "asset": bid,
                "label": b["label"],
                "stats": b["stats"],
                "clip": b["clip"],
                "wavelength": b["wavelength"],
            })

    # Update panel state
    panel["render_id"] = render_id
    panel["dzi"] = dzi_meta
    panel["fmt"] = fmt
    panel["kind"] = result["kind"]
    panel["shape"] = result["shape"]
    panel["dtype"] = result["dtype"]
    panel["stats"] = result["stats"]
    panel["clip"] = result["clip"]
    panel["clip_pct"] = result["clip_pct"]
    panel["log_dr"] = result["log_dr"]
    panel["note"] = result["note"]
    panel["bands_meta"] = band_meta
    panel["n_bands"] = len(band_meta) if band_meta else 0

    # NOTE: we intentionally keep old render directories around. OpenSeadragon
    # may still be loading tiles from the previous render_id (especially the
    # navigator/minimap), and deleting underneath it produces 404s.
    # They're swept when the frame is deleted, and the whole session_<pid>
    # directory is tmp anyway.
    return panel


class WSHub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.clients.discard(ws)

    async def broadcast(self, msg: dict[str, Any]):
        data = json.dumps(msg, default=str)
        dead = []
        async with self._lock:
            clients = list(self.clients)
        for c in clients:
            try:
                await c.send_text(data)
            except Exception:
                dead.append(c)
        if dead:
            async with self._lock:
                for c in dead:
                    self.clients.discard(c)


def build_app(png_dir: Path, load_manifest: dict[str, Any] | None = None) -> Starlette:
    archive = Archive(png_dir)
    hub = WSHub()
    # Pool for CPU-bound rendering so we don't block the event loop
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dbv-render")

    if load_manifest is not None:
        for f in load_manifest.get("frames", []):
            archive.add_frame(dict(f))

    async def run_in_pool(fn, *args):
        return await asyncio.get_event_loop().run_in_executor(executor, fn, *args)

    # ---- routes ----

    async def health(_request: Request):
        return JSONResponse({"ok": True})

    async def post_event(request: Request):
        form = await request.form()
        meta_raw = form.get("metadata")
        if meta_raw is None:
            return JSONResponse({"error": "missing metadata"}, status_code=400)
        meta = json.loads(meta_raw if isinstance(meta_raw, str) else await meta_raw.read())

        # Load raw arrays for each panel
        loaded_arrays: dict[str, np.ndarray] = {}
        for panel in meta.get("panels", []):
            field = form.get(f"raw_{panel['id']}")
            if field is None:
                return JSONResponse(
                    {"error": f"missing raw_{panel['id']}"}, status_code=400)
            raw_bytes = await field.read()
            arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            loaded_arrays[panel["id"]] = arr
            archive.store_raw(panel["id"], arr)

        # Render each panel (off the event loop)
        for panel in meta.get("panels", []):
            await run_in_pool(_render_and_store, archive, panel, loaded_arrays[panel["id"]])

        frame = archive.add_frame(meta)
        await hub.broadcast({"kind": "frame_added", "ts": time.time(), "payload": frame})
        return JSONResponse({"id": frame["id"]})

    async def post_rerender(request: Request):
        pid = request.path_params["panel_id"]
        panel = archive.get_panel(pid)
        if not panel:
            return JSONResponse({"error": "panel not found"}, status_code=404)
        body = await request.json()
        # Apply allowed updates
        for k in ("scale", "clip_pct", "log_dr", "bands", "wavelength", "wavelengths", "kind"):
            if k in body:
                panel[k] = body[k]
        try:
            await run_in_pool(_render_and_store, archive, panel, None)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        await hub.broadcast({
            "kind": "panel_rerendered",
            "ts": time.time(),
            "payload": panel,
        })
        return JSONResponse({"ok": True, "panel": panel})

    async def list_frames(_request: Request):
        return JSONResponse({"frames": archive.list_frames()})

    async def get_dzi_descriptor(request: Request):
        pid = request.path_params["panel_id"]
        rid = request.path_params["render_id"]
        p = archive.dzi_dir / pid / f"{rid}.dzi"
        if not p.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return Response(p.read_text(), media_type="application/xml")

    async def get_dzi_tile(request: Request):
        pid = request.path_params["panel_id"]
        rid = request.path_params["render_id"]
        level = request.path_params["level"]
        tile = request.path_params["tile"]   # "x_y.ext"
        p = archive.dzi_dir / pid / f"{rid}_files" / level / tile
        if not p.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, media_type=_media_for(p.suffix))

    async def get_thumb(request: Request):
        aid = request.path_params["asset_id"]
        p = archive.thumb_path(aid)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, media_type=_media_for(p.suffix))

    async def patch_frame(request: Request):
        fid = request.path_params["frame_id"]
        body = await request.json()
        updated = archive.update_frame(fid, body)
        if updated is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        await hub.broadcast({
            "kind": "frame_updated", "ts": time.time(),
            "payload": {"id": fid, "fields": body},
        })
        return JSONResponse({"ok": True})

    async def patch_panel(request: Request):
        pid = request.path_params["panel_id"]
        body = await request.json()
        updated = archive.update_panel(pid, body)
        if updated is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        await hub.broadcast({
            "kind": "panel_updated", "ts": time.time(),
            "payload": {"id": pid, "fields": body},
        })
        return JSONResponse({"ok": True})

    async def delete_frame(request: Request):
        fid = request.path_params["frame_id"]
        if not archive.delete_frame(fid):
            return JSONResponse({"error": "not found"}, status_code=404)
        await hub.broadcast({"kind": "frame_deleted", "ts": time.time(),
                             "payload": {"id": fid}})
        return JSONResponse({"ok": True})

    async def post_snapshot(request: Request):
        body = await request.json()
        path = Path(os.path.expanduser(body["path"]))
        await run_in_pool(archive.snapshot, path)
        return JSONResponse({"ok": True, "path": str(path)})

    async def ws_endpoint(ws: WebSocket):
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.disconnect(ws)

    routes = [
        Route("/health", health),
        Route("/event", post_event, methods=["POST"]),
        Route("/rerender/{panel_id}", post_rerender, methods=["POST"]),
        Route("/frames", list_frames),
        Route("/dzi/{panel_id}/{render_id}.dzi", get_dzi_descriptor),
        Route("/dzi/{panel_id}/{render_id}_files/{level}/{tile}", get_dzi_tile),
        Route("/thumb/{asset_id}", get_thumb),
        Route("/frames/{frame_id}", patch_frame, methods=["PATCH"]),
        Route("/frames/{frame_id}", delete_frame, methods=["DELETE"]),
        Route("/panels/{panel_id}", patch_panel, methods=["PATCH"]),
        Route("/snapshot", post_snapshot, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
        Mount("/static", app=StaticFiles(directory=str(WEB_DIR)), name="static"),
        Route("/", lambda r: FileResponse(str(WEB_DIR / "index.html"))),
    ]
    app = Starlette(routes=routes)
    app.state.archive = archive
    app.state.hub = hub
    app.state.executor = executor
    return app


def _default_png_dir() -> Path:
    d = SESSION_DIR_BASE / f"session_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_dbv(path: Path) -> tuple[Path, dict[str, Any]]:
    h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    out_dir = SESSION_DIR_BASE / f"loaded_{h}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return out_dir, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args(argv)

    if args.load:
        load_path = Path(os.path.expanduser(args.load))
        if load_path.is_file():
            png_dir, manifest = _load_dbv(load_path)
        else:
            png_dir = load_path
            manifest = json.loads((load_path / "manifest.json").read_text())
        app = build_app(png_dir, load_manifest=manifest)
    else:
        app = build_app(_default_png_dir())

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
