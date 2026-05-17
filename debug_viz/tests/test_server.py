import io
import json
import zipfile

import pytest
from PIL import Image
from starlette.testclient import TestClient

from debug_viz.server import build_app


def _png_bytes(size=(8, 8), color=128):
    img = Image.new("L", size, color).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    app = build_app(tmp_path / "pngs")
    with TestClient(app) as c:
        yield c


def _post_single(client, label="x"):
    meta = {"label": label, "ts": 1.0, "mode": "single",
            "shape": [8, 8], "dtype": "uint8", "stats": {"min": 0, "max": 1},
            "scale": "linear", "clip": [0, 1], "note": None}
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        "full": ("f.png", _png_bytes(), "image/png"),
        "thumb": ("t.png", _png_bytes((4, 4)), "image/png"),
    }
    r = client.post("/event", files=files)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_event_then_frames(client):
    fid = _post_single(client, "hello")
    r = client.get("/frames")
    assert r.status_code == 200
    frames = r.json()["frames"]
    assert len(frames) == 1
    assert frames[0]["id"] == fid
    assert frames[0]["label"] == "hello"
    assert frames[0]["thumb_b64"]


def test_png_endpoint(client):
    fid = _post_single(client)
    r = client.get(f"/png/{fid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_patch_label(client):
    fid = _post_single(client, "orig")
    r = client.patch(f"/frames/{fid}", json={"label": "renamed"})
    assert r.status_code == 200
    r2 = client.get("/frames").json()["frames"][0]
    assert r2["label"] == "renamed"


def test_delete_frame(client, tmp_path):
    fid = _post_single(client)
    png = tmp_path / "pngs" / f"{fid}.png"
    assert png.exists()
    r = client.delete(f"/frames/{fid}")
    assert r.status_code == 200
    assert not png.exists()
    assert client.get("/frames").json()["frames"] == []


def test_ws_broadcast_added(client):
    with client.websocket_connect("/ws") as ws:
        _post_single(client, "ws_test")
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "frame_added"
        assert msg["payload"]["label"] == "ws_test"


def test_ws_broadcast_updated_and_deleted(client):
    with client.websocket_connect("/ws") as ws:
        fid = _post_single(client)
        ws.receive_text()  # frame_added
        client.patch(f"/frames/{fid}", json={"label": "new"})
        m = json.loads(ws.receive_text())
        assert m["kind"] == "frame_updated"
        client.delete(f"/frames/{fid}")
        m = json.loads(ws.receive_text())
        assert m["kind"] == "frame_deleted"


def test_compare_event(client):
    meta = {"label": "cmp", "ts": 1.0, "mode": "compare",
            "shape": [[4, 4], [4, 4]], "dtype": ["uint8", "uint8"],
            "stats": [{"min": 0}, {"min": 1}],
            "scale": "linear", "clip": [[0, 1], [0, 1]], "note": None}
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        "full_a": ("a.png", _png_bytes(), "image/png"),
        "thumb_a": ("at.png", _png_bytes((4, 4)), "image/png"),
        "full_b": ("b.png", _png_bytes(color=200), "image/png"),
        "thumb_b": ("bt.png", _png_bytes((4, 4), color=200), "image/png"),
    }
    r = client.post("/event", files=files)
    assert r.status_code == 200
    frames = client.get("/frames").json()["frames"]
    assert frames[0]["mode"] == "compare"
    assert len(frames[0]["thumb_b64"]) == 2
    assert len(frames[0]["sub_ids"]) == 2


def test_snapshot_roundtrip(client, tmp_path):
    fid = _post_single(client, "snap")
    out = tmp_path / "out.dbv"
    r = client.post("/snapshot", json={"path": str(out)})
    assert r.status_code == 200
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "manifest.json" in names
        assert any(n.startswith("frames/") and n.endswith(".png") for n in names)
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["version"] == 1
        assert manifest["frames"][0]["id"] == fid
        assert manifest["frames"][0]["thumb_b64"]


def test_load_manifest(tmp_path):
    # First app, write a frame and snapshot
    app1 = build_app(tmp_path / "pngs1")
    with TestClient(app1) as c1:
        fid = _post_single(c1, "preserved")
        out = tmp_path / "x.dbv"
        c1.post("/snapshot", json={"path": str(out)})

    # Unzip manually & feed to new app
    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(out) as z:
        z.extractall(extract)
    manifest = json.loads((extract / "manifest.json").read_text())
    app2 = build_app(extract / "frames", load_manifest=manifest)
    with TestClient(app2) as c2:
        frames = c2.get("/frames").json()["frames"]
        assert len(frames) == 1
        assert frames[0]["id"] == fid
        assert frames[0]["label"] == "preserved"
        assert frames[0]["thumb_b64"]
        r = c2.get(f"/png/{fid}")
        assert r.status_code == 200
