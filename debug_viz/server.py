"""Starlette server: asset-based archive + WS broadcast."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

WEB_DIR = Path(__file__).parent / "web"
SESSION_DIR_BASE = Path("/tmp/debug_viz")

# Manifest version. v2 = asset-id storage + kind/bands + JPEG/PNG hybrid.
MANIFEST_VERSION = 2

_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".png": "image/png", ".webp": "image/webp"}


def _media_for(suffix: str) -> str:
    return _MEDIA.get(suffix.lower(), "application/octet-stream")


class Archive:
    """Disk-backed gallery. Assets keyed by asset_id; frames reference asset_ids.

    Layout under png_dir:
        full/<asset_id>.<ext>
        thumb/<asset_id>.<ext>
    """

    def __init__(self, png_dir: Path):
        self.png_dir = png_dir
        self.full_dir = png_dir / "full"
        self.thumb_dir = png_dir / "thumb"
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict[str, Any]] = []

    def _write(self, asset_id: str, ext: str, full: bytes, thumb: bytes) -> None:
        (self.full_dir / f"{asset_id}.{ext}").write_bytes(full)
        (self.thumb_dir / f"{asset_id}.{ext}").write_bytes(thumb)

    def _remove_asset(self, asset_id: str) -> None:
        for d in (self.full_dir, self.thumb_dir):
            for p in d.glob(f"{asset_id}.*"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    def add(self, frame: dict[str, Any], main_full: bytes, main_thumb: bytes,
            bands: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        fid = frame.get("id") or uuid.uuid4().hex
        frame["id"] = fid
        ext = frame.get("fmt", "jpeg")
        self._write(fid, ext, main_full, main_thumb)
        frame["main_asset"] = fid

        if bands:
            band_records = []
            for i, b in enumerate(bands):
                aid = f"{fid}__b{i}"
                self._write(aid, ext, b["full"], b["thumb"])
                band_records.append({
                    "asset": aid,
                    "label": b.get("label", f"ch {i}"),
                    "stats": b.get("stats", {}),
                    "clip": b.get("clip", [0.0, 1.0]),
                })
            frame["bands"] = band_records

        self.frames.append(frame)
        return frame

    def add_compare(self, frame: dict[str, Any], a_full: bytes, a_thumb: bytes,
                    b_full: bytes, b_thumb: bytes, fmt: str = "jpeg") -> dict[str, Any]:
        fid = frame.get("id") or uuid.uuid4().hex
        frame["id"] = fid
        a_id, b_id = f"{fid}__a", f"{fid}__b"
        self._write(a_id, fmt, a_full, a_thumb)
        self._write(b_id, fmt, b_full, b_thumb)
        frame["fmt"] = fmt
        frame["compare_assets"] = [a_id, b_id]
        self.frames.append(frame)
        return frame

    def update(self, fid: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        for f in self.frames:
            if f["id"] == fid:
                if "label" in fields:
                    f["label"] = fields["label"]
                return f
        return None

    def delete(self, fid: str) -> bool:
        for i, f in enumerate(self.frames):
            if f["id"] == fid:
                assets = self._frame_assets(f)
                for aid in assets:
                    self._remove_asset(aid)
                del self.frames[i]
                return True
        return False

    @staticmethod
    def _frame_assets(frame: dict[str, Any]) -> list[str]:
        out = []
        if frame.get("main_asset"):
            out.append(frame["main_asset"])
        out.extend(frame.get("compare_assets") or [])
        out.extend(b["asset"] for b in (frame.get("bands") or []))
        return out

    def asset_path(self, kind: str, asset_id: str) -> Path | None:
        base = self.full_dir if kind == "full" else self.thumb_dir
        matches = list(base.glob(f"{asset_id}.*"))
        return matches[0] if matches else None

    def list_frames(self) -> list[dict[str, Any]]:
        return [dict(f) for f in self.frames]

    def snapshot(self, out_path: Path) -> None:
        manifest = {
            "version": MANIFEST_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frames": self.list_frames(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in self.frames:
                for aid in self._frame_assets(f):
                    fp = self.asset_path("full", aid)
                    tp = self.asset_path("thumb", aid)
                    if fp:
                        z.write(fp, arcname=f"full/{fp.name}")
                    if tp:
                        z.write(tp, arcname=f"thumb/{tp.name}")
            z.writestr("manifest.json", json.dumps(manifest, indent=2))


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

    if load_manifest is not None:
        for f in load_manifest.get("frames", []):
            archive.frames.append(dict(f))

    async def health(_request: Request):
        return JSONResponse({"ok": True})

    async def post_event(request: Request):
        form = await request.form()
        meta_raw = form.get("metadata")
        if meta_raw is None:
            return JSONResponse({"error": "missing metadata"}, status_code=400)
        meta = json.loads(meta_raw if isinstance(meta_raw, str) else await meta_raw.read())
        mode = meta.get("mode", "single")
        if mode == "compare":
            payload = archive.add_compare(
                meta,
                await form["full_a"].read(), await form["thumb_a"].read(),
                await form["full_b"].read(), await form["thumb_b"].read(),
                fmt=meta.get("fmt", "jpeg"),
            )
        else:
            main_full = await form["main_full"].read()
            main_thumb = await form["main_thumb"].read()
            n_bands = int(meta.get("n_bands", 0))
            bands = []
            for i in range(n_bands):
                bands.append({
                    "full": await form[f"band_{i}_full"].read(),
                    "thumb": await form[f"band_{i}_thumb"].read(),
                    "label": (meta.get("band_meta") or [{}] * n_bands)[i].get("label", f"ch {i}"),
                    "stats": (meta.get("band_meta") or [{}] * n_bands)[i].get("stats", {}),
                    "clip": (meta.get("band_meta") or [{}] * n_bands)[i].get("clip", [0.0, 1.0]),
                })
            payload = archive.add(meta, main_full, main_thumb, bands=bands or None)
        await hub.broadcast({"kind": "frame_added", "ts": time.time(), "payload": payload})
        return JSONResponse({"id": payload["id"]})

    async def list_frames(_request: Request):
        return JSONResponse({"frames": archive.list_frames()})

    async def get_full(request: Request):
        aid = request.path_params["asset_id"]
        p = archive.asset_path("full", aid)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, media_type=_media_for(p.suffix))

    async def get_thumb(request: Request):
        aid = request.path_params["asset_id"]
        p = archive.asset_path("thumb", aid)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, media_type=_media_for(p.suffix))

    async def patch_frame(request: Request):
        fid = request.path_params["frame_id"]
        body = await request.json()
        updated = archive.update(fid, body)
        if updated is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        await hub.broadcast({
            "kind": "frame_updated", "ts": time.time(),
            "payload": {"id": fid, "fields": body},
        })
        return JSONResponse({"ok": True})

    async def delete_frame(request: Request):
        fid = request.path_params["frame_id"]
        if not archive.delete(fid):
            return JSONResponse({"error": "not found"}, status_code=404)
        await hub.broadcast({"kind": "frame_deleted", "ts": time.time(), "payload": {"id": fid}})
        return JSONResponse({"ok": True})

    async def post_snapshot(request: Request):
        body = await request.json()
        path = Path(os.path.expanduser(body["path"]))
        archive.snapshot(path)
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
        Route("/frames", list_frames),
        Route("/image/{asset_id}", get_full),
        Route("/thumb/{asset_id}", get_thumb),
        Route("/frames/{frame_id}", patch_frame, methods=["PATCH"]),
        Route("/frames/{frame_id}", delete_frame, methods=["DELETE"]),
        Route("/snapshot", post_snapshot, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
        Mount("/static", app=StaticFiles(directory=str(WEB_DIR)), name="static"),
        Route("/", lambda r: FileResponse(str(WEB_DIR / "index.html"))),
    ]
    app = Starlette(routes=routes)
    app.state.archive = archive
    app.state.hub = hub
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
