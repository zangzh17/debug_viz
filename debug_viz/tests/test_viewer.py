import io
import json
import zipfile

from PIL import Image
from starlette.testclient import TestClient

from debug_viz.server import build_app, _load_dbv


def _make_dbv(path, frames):
    with zipfile.ZipFile(path, "w") as z:
        manifest = {"version": 1, "created_at": "now", "frames": frames}
        z.writestr("manifest.json", json.dumps(manifest))
        for f in frames:
            if f["mode"] == "compare":
                for sid in f["sub_ids"]:
                    buf = io.BytesIO()
                    Image.new("RGB", (4, 4), (200, 100, 50)).save(buf, "PNG")
                    z.writestr(f"frames/{sid}.png", buf.getvalue())
            else:
                buf = io.BytesIO()
                Image.new("RGB", (4, 4), (100, 200, 50)).save(buf, "PNG")
                z.writestr(f"frames/{f['id']}.png", buf.getvalue())


def test_load_dbv_and_serve(tmp_path):
    dbv = tmp_path / "test.dbv"
    frames = [
        {"id": "abc", "label": "one", "mode": "single", "shape": [4, 4],
         "dtype": "uint8", "stats": {"min": 0, "max": 1},
         "scale": "linear", "clip": [0, 1], "note": None,
         "thumb_b64": "AAAA", "full_png": "frames/abc.png"},
        {"id": "xyz", "label": "cmp", "mode": "compare",
         "sub_ids": ["xyz_a", "xyz_b"],
         "shape": [[4, 4], [4, 4]], "dtype": ["uint8", "uint8"],
         "stats": [{}, {}], "scale": "linear", "clip": [[0, 1], [0, 1]],
         "note": None, "thumb_b64": ["AAAA", "BBBB"],
         "full_png": ["frames/xyz_a.png", "frames/xyz_b.png"]},
    ]
    _make_dbv(dbv, frames)

    png_dir, manifest = _load_dbv(dbv)
    assert (png_dir / "abc.png").exists()
    assert (png_dir / "xyz_a.png").exists()

    app = build_app(png_dir, load_manifest=manifest)
    with TestClient(app) as c:
        out = c.get("/frames").json()["frames"]
        assert len(out) == 2
        ids = {f["id"] for f in out}
        assert ids == {"abc", "xyz"}
        r = c.get("/png/abc")
        assert r.status_code == 200
        r2 = c.get("/png/xyz_a")
        assert r2.status_code == 200
