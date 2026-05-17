"""Starlette server: gallery archive + WS broadcast."""
from __future__ import annotations

import argparse
import asyncio
import base64
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
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

WEB_DIR = Path(__file__).parent / "web"
SESSION_DIR_BASE = Path("/tmp/debug_viz")


class Archive:
    def __init__(self, png_dir: Path):
        self.png_dir = png_dir
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict[str, Any]] = []
        self.thumbs_b64: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def add(self, frame: dict[str, Any], full_png: bytes, thumb_png: bytes) -> dict[str, Any]:
        fid = frame.get("id") or uuid.uuid4().hex
        frame["id"] = fid
        png_path = self.png_dir / f"{fid}.png"
        png_path.write_bytes(full_png)
        frame["full_png"] = f"frames/{fid}.png"
        thumb_b64 = base64.b64encode(thumb_png).decode("ascii")
        self.thumbs_b64[fid] = thumb_b64
        record = dict(frame)
        self.frames.append(record)
        return {**record, "thumb_b64": thumb_b64}

    def add_compare(self, frame: dict[str, Any], a_full: bytes, a_thumb: bytes,
                    b_full: bytes, b_thumb: bytes) -> dict[str, Any]:
        fid = frame.get("id") or uuid.uuid4().hex
        frame["id"] = fid
        a_id, b_id = f"{fid}_a", f"{fid}_b"
        (self.png_dir / f"{a_id}.png").write_bytes(a_full)
        (self.png_dir / f"{b_id}.png").write_bytes(b_full)
        frame["full_png"] = [f"frames/{a_id}.png", f"frames/{b_id}.png"]
        a_b64 = base64.b64encode(a_thumb).decode("ascii")
        b_b64 = base64.b64encode(b_thumb).decode("ascii")
        self.thumbs_b64[a_id] = a_b64
        self.thumbs_b64[b_id] = b_b64
        frame["sub_ids"] = [a_id, b_id]
        record = dict(frame)
        self.frames.append(record)
        return {**record, "thumb_b64": [a_b64, b_b64]}

    def get_with_thumbs(self) -> list[dict[str, Any]]:
        out = []
        for f in self.frames:
            r = dict(f)
            if f.get("mode") == "compare":
                sub_ids = f.get("sub_ids") or []
                r["thumb_b64"] = [self.thumbs_b64.get(sid, "") for sid in sub_ids]
            else:
                r["thumb_b64"] = self.thumbs_b64.get(f["id"], "")
            out.append(r)
        return out

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
                if f.get("mode") == "compare":
                    for sid in f.get("sub_ids", []):
                        p = self.png_dir / f"{sid}.png"
                        if p.exists():
                            p.unlink()
                        self.thumbs_b64.pop(sid, None)
                else:
                    p = self.png_dir / f"{fid}.png"
                    if p.exists():
                        p.unlink()
                    self.thumbs_b64.pop(fid, None)
                del self.frames[i]
                return True
        return False

    def png_path(self, fid: str) -> Path | None:
        p = self.png_dir / f"{fid}.png"
        return p if p.exists() else None

    def snapshot(self, out_path: Path) -> None:
        manifest = {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frames": [],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in self.frames:
                rec = dict(f)
                if f.get("mode") == "compare":
                    sub_ids = f.get("sub_ids", [])
                    rec["thumb_b64"] = [self.thumbs_b64.get(sid, "") for sid in sub_ids]
                    rec["full_png"] = [f"frames/{sid}.png" for sid in sub_ids]
                    for sid in sub_ids:
                        p = self.png_dir / f"{sid}.png"
                        if p.exists():
                            z.write(p, arcname=f"frames/{sid}.png")
                else:
                    fid = f["id"]
                    rec["thumb_b64"] = self.thumbs_b64.get(fid, "")
                    rec["full_png"] = f"frames/{fid}.png"
                    p = self.png_dir / f"{fid}.png"
                    if p.exists():
                        z.write(p, arcname=f"frames/{fid}.png")
                manifest["frames"].append(rec)
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
            rec = {k: v for k, v in f.items() if k not in ("thumb_b64",)}
            thumb = f.get("thumb_b64")
            if rec.get("mode") == "compare":
                if isinstance(thumb, list) and rec.get("sub_ids"):
                    for sid, t in zip(rec["sub_ids"], thumb):
                        archive.thumbs_b64[sid] = t
            else:
                if isinstance(thumb, str):
                    archive.thumbs_b64[rec["id"]] = thumb
            archive.frames.append(rec)

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
            a_full = await form["full_a"].read()
            a_thumb = await form["thumb_a"].read()
            b_full = await form["full_b"].read()
            b_thumb = await form["thumb_b"].read()
            payload = archive.add_compare(meta, a_full, a_thumb, b_full, b_thumb)
        else:
            full_png = await form["full"].read()
            thumb_png = await form["thumb"].read()
            payload = archive.add(meta, full_png, thumb_png)
        await hub.broadcast({"kind": "frame_added", "ts": time.time(), "payload": payload})
        return JSONResponse({"id": payload["id"]})

    async def list_frames(_request: Request):
        return JSONResponse({"frames": archive.get_with_thumbs()})

    async def get_png(request: Request):
        fid = request.path_params["frame_id"]
        p = archive.png_path(fid)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p, media_type="image/png")

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
        ok = archive.delete(fid)
        if not ok:
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
        Route("/png/{frame_id}", get_png),
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
    pid = os.getpid()
    d = SESSION_DIR_BASE / f"session_{pid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_dbv(path: Path) -> tuple[Path, dict[str, Any]]:
    """Unzip .dbv to a tmp dir, return (png_dir, manifest)."""
    import hashlib
    h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    out_dir = SESSION_DIR_BASE / f"loaded_{h}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return out_dir / "frames", manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--load", type=str, default=None,
                        help="Path to .dbv or extracted dir to preload")
    args = parser.parse_args(argv)

    if args.load:
        load_path = Path(os.path.expanduser(args.load))
        if load_path.is_file():
            png_dir, manifest = _load_dbv(load_path)
        else:
            png_dir = load_path / "frames"
            manifest = json.loads((load_path / "manifest.json").read_text())
        app = build_app(png_dir, load_manifest=manifest)
    else:
        app = build_app(_default_png_dir())

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
