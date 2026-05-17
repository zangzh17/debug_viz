import io
import json
import zipfile

import pytest
from PIL import Image
from starlette.testclient import TestClient

from debug_viz.server import build_app, MANIFEST_VERSION


def _img_bytes(size=(8, 8), color=128, fmt="JPEG"):
    img = Image.new("L", size, color).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    app = build_app(tmp_path / "assets")
    with TestClient(app) as c:
        yield c


def _post_single(client, label="x", kind="gray"):
    meta = {"label": label, "ts": 1.0, "mode": "single", "kind": kind,
            "fmt": "jpeg", "shape": [8, 8], "dtype": "uint8",
            "stats": {"min": 0, "max": 1}, "scale": "linear",
            "clip": [0, 1], "note": None, "n_bands": 0}
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        "main_full": ("f.jpeg", _img_bytes(), "image/jpeg"),
        "main_thumb": ("t.jpeg", _img_bytes((4, 4)), "image/jpeg"),
    }
    r = client.post("/event", files=files)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _post_multi(client, label="m", n_bands=3):
    band_meta = [{"label": f"ch {i}", "stats": {"min": 0}, "clip": [0, 1]}
                 for i in range(n_bands)]
    meta = {"label": label, "ts": 1.0, "mode": "single", "kind": "multi",
            "fmt": "jpeg", "shape": [8, 8, n_bands], "dtype": "float32",
            "stats": {}, "scale": "linear", "clip": [0, 1], "note": None,
            "n_bands": n_bands, "band_meta": band_meta}
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        "main_full": ("f.jpeg", _img_bytes(), "image/jpeg"),
        "main_thumb": ("t.jpeg", _img_bytes((4, 4)), "image/jpeg"),
    }
    for i in range(n_bands):
        files[f"band_{i}_full"] = (f"b{i}.jpeg", _img_bytes(color=50 + 20 * i), "image/jpeg")
        files[f"band_{i}_thumb"] = (f"b{i}t.jpeg", _img_bytes((4, 4), color=50 + 20 * i), "image/jpeg")
    r = client.post("/event", files=files)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_health(client):
    assert client.get("/health").json()["ok"] is True


def test_event_then_frames(client):
    fid = _post_single(client, "hello")
    frames = client.get("/frames").json()["frames"]
    assert len(frames) == 1
    f = frames[0]
    assert f["id"] == fid
    assert f["label"] == "hello"
    assert f["kind"] == "gray"
    assert f["main_asset"] == fid
    assert "thumb_b64" not in f  # No longer inlined


def test_image_endpoint(client):
    fid = _post_single(client)
    r = client.get(f"/image/{fid}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_thumb_endpoint(client):
    fid = _post_single(client)
    r = client.get(f"/thumb/{fid}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_patch_label(client):
    fid = _post_single(client, "orig")
    r = client.patch(f"/frames/{fid}", json={"label": "renamed"})
    assert r.status_code == 200
    assert client.get("/frames").json()["frames"][0]["label"] == "renamed"


def test_delete_frame(client, tmp_path):
    fid = _post_single(client)
    full = tmp_path / "assets" / "full" / f"{fid}.jpeg"
    thumb = tmp_path / "assets" / "thumb" / f"{fid}.jpeg"
    assert full.exists() and thumb.exists()
    client.delete(f"/frames/{fid}")
    assert not full.exists()
    assert not thumb.exists()
    assert client.get("/frames").json()["frames"] == []


def test_ws_broadcast(client):
    with client.websocket_connect("/ws") as ws:
        fid = _post_single(client, "ws_test")
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "frame_added"
        assert msg["payload"]["label"] == "ws_test"
        client.patch(f"/frames/{fid}", json={"label": "new"})
        assert json.loads(ws.receive_text())["kind"] == "frame_updated"
        client.delete(f"/frames/{fid}")
        assert json.loads(ws.receive_text())["kind"] == "frame_deleted"


def test_compare_event(client):
    meta = {"label": "cmp", "ts": 1.0, "mode": "compare", "kind": "gray",
            "fmt": "jpeg", "shape": [[4, 4], [4, 4]], "dtype": ["uint8", "uint8"],
            "stats": [{"min": 0}, {"min": 1}], "scale": "linear",
            "clip": [[0, 1], [0, 1]], "note": None}
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        "full_a": ("a.jpeg", _img_bytes(), "image/jpeg"),
        "thumb_a": ("at.jpeg", _img_bytes((4, 4)), "image/jpeg"),
        "full_b": ("b.jpeg", _img_bytes(color=200), "image/jpeg"),
        "thumb_b": ("bt.jpeg", _img_bytes((4, 4), color=200), "image/jpeg"),
    }
    r = client.post("/event", files=files)
    assert r.status_code == 200
    frame = client.get("/frames").json()["frames"][0]
    assert frame["mode"] == "compare"
    assert len(frame["compare_assets"]) == 2
    for aid in frame["compare_assets"]:
        assert client.get(f"/image/{aid}").status_code == 200
        assert client.get(f"/thumb/{aid}").status_code == 200


def test_multi_event(client, tmp_path):
    fid = _post_multi(client, n_bands=4)
    f = client.get("/frames").json()["frames"][0]
    assert f["kind"] == "multi"
    assert len(f["bands"]) == 4
    for i, b in enumerate(f["bands"]):
        assert b["asset"] == f"{fid}__b{i}"
        assert b["label"] == f"ch {i}"
        # Each band asset must be servable
        assert client.get(f"/image/{b['asset']}").status_code == 200
        assert client.get(f"/thumb/{b['asset']}").status_code == 200


def test_multi_delete_cleans_bands(client, tmp_path):
    fid = _post_multi(client, n_bands=3)
    full_dir = tmp_path / "assets" / "full"
    assert (full_dir / f"{fid}.jpeg").exists()
    for i in range(3):
        assert (full_dir / f"{fid}__b{i}.jpeg").exists()
    client.delete(f"/frames/{fid}")
    assert not (full_dir / f"{fid}.jpeg").exists()
    for i in range(3):
        assert not (full_dir / f"{fid}__b{i}.jpeg").exists()


def test_snapshot_roundtrip(client, tmp_path):
    fid = _post_single(client, "snap")
    out = tmp_path / "out.dbv"
    client.post("/snapshot", json={"path": str(out)})
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "manifest.json" in names
        assert f"full/{fid}.jpeg" in names
        assert f"thumb/{fid}.jpeg" in names
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["version"] == MANIFEST_VERSION
        assert manifest["frames"][0]["id"] == fid


def test_snapshot_includes_bands(client, tmp_path):
    fid = _post_multi(client, n_bands=4)
    out = tmp_path / "m.dbv"
    client.post("/snapshot", json={"path": str(out)})
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        for i in range(4):
            assert f"full/{fid}__b{i}.jpeg" in names
            assert f"thumb/{fid}__b{i}.jpeg" in names


def test_load_manifest(tmp_path):
    app1 = build_app(tmp_path / "a1")
    with TestClient(app1) as c1:
        fid = _post_multi(c1, "preserved", n_bands=2)
        out = tmp_path / "x.dbv"
        c1.post("/snapshot", json={"path": str(out)})

    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(out) as z:
        z.extractall(extract)
    manifest = json.loads((extract / "manifest.json").read_text())
    app2 = build_app(extract, load_manifest=manifest)
    with TestClient(app2) as c2:
        frames = c2.get("/frames").json()["frames"]
        assert len(frames) == 1
        assert frames[0]["id"] == fid
        assert frames[0]["kind"] == "multi"
        assert len(frames[0]["bands"]) == 2
        # All band image and thumb endpoints work in replay
        for b in frames[0]["bands"]:
            assert c2.get(f"/image/{b['asset']}").status_code == 200
            assert c2.get(f"/thumb/{b['asset']}").status_code == 200
