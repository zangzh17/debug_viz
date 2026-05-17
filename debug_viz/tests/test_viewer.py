import io
import json
import zipfile

from PIL import Image
from starlette.testclient import TestClient

from debug_viz.server import build_app, MANIFEST_VERSION, _load_dbv


def _webp(color=128, size=(8, 8)):
    img = Image.new("L", size, color).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_dbv(path, frames):
    with zipfile.ZipFile(path, "w") as z:
        manifest = {"version": MANIFEST_VERSION, "created_at": "now", "frames": frames}
        z.writestr("manifest.json", json.dumps(manifest))
        # write a webp for every asset referenced
        seen = set()

        def write(aid):
            if aid in seen:
                return
            seen.add(aid)
            z.writestr(f"full/{aid}.jpeg", _webp())
            z.writestr(f"thumb/{aid}.jpeg", _webp(size=(4, 4)))

        for f in frames:
            if f.get("main_asset"):
                write(f["main_asset"])
            for aid in f.get("compare_assets", []):
                write(aid)
            for b in f.get("bands", []):
                write(b["asset"])


def test_load_dbv_and_serve(tmp_path):
    dbv = tmp_path / "test.dbv"
    frames = [
        {"id": "abc", "label": "one", "mode": "single", "kind": "gray",
         "fmt": "jpeg", "shape": [8, 8], "dtype": "uint8",
         "stats": {"min": 0, "max": 1}, "scale": "linear",
         "clip": [0, 1], "note": None, "main_asset": "abc"},
        {"id": "xyz", "label": "cmp", "mode": "compare", "kind": "gray",
         "fmt": "jpeg", "compare_assets": ["xyz__a", "xyz__b"],
         "shape": [[8, 8], [8, 8]], "dtype": ["uint8", "uint8"],
         "stats": [{}, {}], "scale": "linear", "clip": [[0, 1], [0, 1]],
         "note": None},
        {"id": "mmm", "label": "multi", "mode": "single", "kind": "multi",
         "fmt": "jpeg", "main_asset": "mmm", "shape": [8, 8, 3],
         "dtype": "float32", "stats": {}, "scale": "linear",
         "clip": [0, 1], "note": "3 channels",
         "bands": [
             {"asset": "mmm__b0", "label": "ch 0", "stats": {}, "clip": [0, 1]},
             {"asset": "mmm__b1", "label": "ch 1", "stats": {}, "clip": [0, 1]},
             {"asset": "mmm__b2", "label": "ch 2", "stats": {}, "clip": [0, 1]},
         ]},
    ]
    _make_dbv(dbv, frames)

    base, manifest = _load_dbv(dbv)
    assert (base / "full" / "abc.jpeg").exists()
    assert (base / "full" / "mmm__b1.jpeg").exists()

    app = build_app(base, load_manifest=manifest)
    with TestClient(app) as c:
        out = c.get("/frames").json()["frames"]
        assert len(out) == 3
        ids = {f["id"] for f in out}
        assert ids == {"abc", "xyz", "mmm"}

        # Single
        assert c.get("/image/abc").status_code == 200
        assert c.get("/thumb/abc").status_code == 200
        # Compare assets
        assert c.get("/image/xyz__a").status_code == 200
        assert c.get("/image/xyz__b").status_code == 200
        # Multi composite + bands
        assert c.get("/image/mmm").status_code == 200
        for i in range(3):
            assert c.get(f"/image/mmm__b{i}").status_code == 200
            assert c.get(f"/thumb/mmm__b{i}").status_code == 200
