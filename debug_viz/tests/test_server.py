"""Server: raw-arrays-in, DZI-out, re-render, snapshot round-trip."""
import io
import json
import uuid
import zipfile

import numpy as np
import pytest
from starlette.testclient import TestClient

from debug_viz.server import MANIFEST_VERSION, build_app


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _single(arr, label="x", *, kind="auto", scale="linear",
            clip_pct=(1.0, 99.0), wavelength=None, wavelengths=None,
            bands=None):
    pid = uuid.uuid4().hex[:12]
    meta = {
        "id": uuid.uuid4().hex[:12],
        "label": label,
        "ts": 1.0,
        "mode": "single",
        "panels": [{
            "id": pid, "label": "",
            "kind": kind, "scale": scale,
            "clip_pct": list(clip_pct),
            "bands": list(bands) if bands else None,
            "wavelength": wavelength,
            "wavelengths": list(wavelengths) if wavelengths else None,
        }],
    }
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        f"raw_{pid}": (f"{pid}.npy", _npy(arr), "application/octet-stream"),
    }
    return meta, files, pid


@pytest.fixture
def client(tmp_path):
    app = build_app(tmp_path / "assets")
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["ok"] is True


def test_event_creates_dzi_and_thumb(client, tmp_path):
    arr = np.random.default_rng(0).random((100, 200), dtype=np.float32)
    _, files, pid = _single(arr, label="g")
    r = client.post("/event", files=files)
    assert r.status_code == 200, r.text
    fid = r.json()["id"]

    frame = client.get("/frames").json()["frames"][0]
    assert frame["id"] == fid
    panel = frame["panels"][0]
    assert panel["id"] == pid
    assert panel["kind"] == "gray"
    assert panel["dzi"]["width"] == 200 and panel["dzi"]["height"] == 100
    assert panel["render_id"]

    # DZI descriptor + at least one bottom-level tile must be served
    rid = panel["render_id"]
    desc = client.get(f"/dzi/{pid}/{rid}.dzi")
    assert desc.status_code == 200
    assert b"<Image" in desc.content
    ext = "jpg" if panel["fmt"] == "jpeg" else panel["fmt"]
    tile = client.get(f"/dzi/{pid}/{rid}_files/0/0_0.{ext}")
    assert tile.status_code == 200

    thumb = client.get(f"/thumb/{pid}")
    assert thumb.status_code == 200


def test_event_with_wavelength_for_mono(client):
    arr = np.full((20, 20), 0.8, dtype=np.float32)
    _, files, pid = _single(arr, kind="mono", wavelength=656.3)
    client.post("/event", files=files)
    panel = client.get("/frames").json()["frames"][0]["panels"][0]
    assert panel["wavelength"] == 656.3
    assert "656" in (panel.get("note") or "")


def test_event_multi_with_wavelengths_generates_band_thumbs(client):
    arr = np.random.default_rng(0).random((30, 30, 4), dtype=np.float32)
    _, files, pid = _single(arr, kind="multi", wavelengths=[450, 550, 650, 750])
    client.post("/event", files=files)
    panel = client.get("/frames").json()["frames"][0]["panels"][0]
    assert panel["kind"] == "multi"
    assert panel["n_bands"] == 4
    # Per-band thumbs must be servable
    for bm in panel["bands_meta"]:
        r = client.get(f"/thumb/{bm['asset']}")
        assert r.status_code == 200


def test_compare_three_panels(client):
    a = np.full((16, 16), 0.3, dtype=np.float32)
    b = np.full((16, 16), 0.6, dtype=np.float32)
    c = np.full((16, 16), 0.9, dtype=np.float32)
    panels = []
    files = {}
    for arr, sub in zip([a, b, c], ["noisy", "denoised", "gt"]):
        pid = uuid.uuid4().hex[:12]
        panels.append({"id": pid, "label": sub, "kind": "auto",
                       "scale": "linear", "clip_pct": [1.0, 99.0],
                       "bands": None, "wavelength": None, "wavelengths": None})
        files[f"raw_{pid}"] = (f"{pid}.npy", _npy(arr), "application/octet-stream")
    meta = {"id": uuid.uuid4().hex[:12], "label": "3way", "ts": 1.0,
            "mode": "compare", "panels": panels}
    files["metadata"] = (None, json.dumps(meta), "application/json")
    r = client.post("/event", files=files)
    assert r.status_code == 200
    f = client.get("/frames").json()["frames"][0]
    assert f["mode"] == "compare"
    assert [p["label"] for p in f["panels"]] == ["noisy", "denoised", "gt"]
    # Each panel has its own DZI
    for p in f["panels"]:
        assert p["dzi"]["width"] == 16


def test_rerender_changes_render_id_and_dzi(client):
    arr = np.random.default_rng(0).random((50, 50), dtype=np.float32)
    _, files, pid = _single(arr)
    client.post("/event", files=files)
    rid_before = client.get("/frames").json()["frames"][0]["panels"][0]["render_id"]

    r = client.post(f"/rerender/{pid}", json={"scale": "log", "clip_pct": [5.0, 95.0]})
    assert r.status_code == 200
    panel = r.json()["panel"]
    assert panel["render_id"] != rid_before
    assert panel["scale"] == "log"
    assert panel["clip_pct"] == [5.0, 95.0]

    # New DZI accessible
    new_desc = client.get(f"/dzi/{pid}/{panel['render_id']}.dzi")
    assert new_desc.status_code == 200
    # Old DZI intentionally kept (OSD navigator may still be loading from it).
    # It's removed only when the frame itself is deleted.
    old_desc = client.get(f"/dzi/{pid}/{rid_before}.dzi")
    assert old_desc.status_code == 200


def test_rerender_to_single_band_mono_view(client):
    """Click a band in the strip → re-render as kind=mono, bands=[i]."""
    arr = np.random.default_rng(0).random((20, 20, 4), dtype=np.float32)
    arr[..., 1] = 0.9  # band 1 bright
    _, files, pid = _single(arr, kind="multi", wavelengths=[450, 550, 650, 750])
    client.post("/event", files=files)

    r = client.post(f"/rerender/{pid}", json={
        "kind": "mono", "bands": [1], "wavelength": 550, "wavelengths": None,
    })
    assert r.status_code == 200
    p = r.json()["panel"]
    assert p["kind"] == "mono"
    assert p["wavelength"] == 550


def test_patch_frame_label(client):
    arr = np.zeros((4, 4), dtype=np.float32)
    _, files, _ = _single(arr, label="orig")
    client.post("/event", files=files)
    fid = client.get("/frames").json()["frames"][0]["id"]
    r = client.patch(f"/frames/{fid}", json={"label": "renamed"})
    assert r.status_code == 200
    assert client.get("/frames").json()["frames"][0]["label"] == "renamed"


def test_patch_panel_label(client):
    arr = np.zeros((4, 4), dtype=np.float32)
    _, files, pid = _single(arr)
    client.post("/event", files=files)
    r = client.patch(f"/panels/{pid}", json={"label": "before"})
    assert r.status_code == 200
    assert client.get("/frames").json()["frames"][0]["panels"][0]["label"] == "before"


def test_delete_frame_removes_dzi_and_raw(client, tmp_path):
    arr = np.zeros((4, 4), dtype=np.float32)
    _, files, pid = _single(arr)
    client.post("/event", files=files)
    fid = client.get("/frames").json()["frames"][0]["id"]
    raw = tmp_path / "assets" / "raw" / f"{pid}.npy"
    dzi_dir = tmp_path / "assets" / "dzi" / pid
    assert raw.exists() and dzi_dir.exists()
    client.delete(f"/frames/{fid}")
    assert not raw.exists()
    assert not dzi_dir.exists()


def test_ws_broadcast_added_and_rerendered(client):
    arr = np.zeros((4, 4), dtype=np.float32)
    _, files, pid = _single(arr)
    with client.websocket_connect("/ws") as ws:
        client.post("/event", files=files)
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "frame_added"
        client.post(f"/rerender/{pid}", json={"scale": "log"})
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "panel_rerendered"
        assert msg["payload"]["id"] == pid


def test_snapshot_roundtrip(client, tmp_path):
    arr = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    _, files, pid = _single(arr, label="snap")
    client.post("/event", files=files)
    fid = client.get("/frames").json()["frames"][0]["id"]
    rid = client.get("/frames").json()["frames"][0]["panels"][0]["render_id"]

    out = tmp_path / "out.dbv"
    client.post("/snapshot", json={"path": str(out)})
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert "manifest.json" in names
        assert f"raw/{pid}.npy" in names
        assert any(n.startswith(f"thumb/{pid}.") for n in names)
        assert any(n.startswith(f"dzi/{pid}/{rid}_files/") for n in names)
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["version"] == MANIFEST_VERSION
        assert manifest["frames"][0]["id"] == fid


def test_replay_from_snapshot(tmp_path):
    """Save → extract → load_manifest → DZI / thumb endpoints still work."""
    arr = np.random.default_rng(0).random((20, 20), dtype=np.float32)
    app1 = build_app(tmp_path / "a1")
    with TestClient(app1) as c1:
        _, files, pid = _single(arr, label="x")
        c1.post("/event", files=files)
        rid = c1.get("/frames").json()["frames"][0]["panels"][0]["render_id"]
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
        assert len(frames) == 1 and frames[0]["panels"][0]["id"] == pid
        # DZI tiles and raw both survive
        assert c2.get(f"/dzi/{pid}/{rid}.dzi").status_code == 200
        assert c2.get(f"/thumb/{pid}").status_code == 200


def test_replay_supports_rerender(tmp_path):
    """After loading from snapshot, re-render must still work (raw was saved)."""
    arr = np.random.default_rng(0).random((20, 20), dtype=np.float32)
    app1 = build_app(tmp_path / "a1")
    with TestClient(app1) as c1:
        _, files, pid = _single(arr)
        c1.post("/event", files=files)
        out = tmp_path / "x.dbv"
        c1.post("/snapshot", json={"path": str(out)})

    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(out) as z:
        z.extractall(extract)
    manifest = json.loads((extract / "manifest.json").read_text())

    app2 = build_app(extract, load_manifest=manifest)
    with TestClient(app2) as c2:
        r = c2.post(f"/rerender/{pid}", json={"scale": "log"})
        assert r.status_code == 200, r.text
        assert r.json()["panel"]["scale"] == "log"
