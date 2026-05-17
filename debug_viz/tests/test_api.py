import io
import json
from unittest.mock import patch

import numpy as np

from debug_viz import api


def _parse_meta(files):
    return json.loads(files["metadata"][1])


def test_sanitize_nan():
    out = api._sanitize_for_json({"a": float("nan"), "b": [1.0, float("inf"), 2.0]})
    assert out["a"] is None
    assert out["b"] == [1.0, None, 2.0]


def test_view_sends_metadata_and_one_raw(gray_2d):
    posts = []

    def fake_post(url, files=None, **kw):
        posts.append((url, _parse_meta(files), set(files.keys())))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_2d, label="foo")

    assert len(posts) == 1
    url, meta, keys = posts[0]
    assert url.endswith("/event")
    assert meta["mode"] == "single"
    assert meta["label"] == "foo"
    assert len(meta["panels"]) == 1
    panel = meta["panels"][0]
    # Raw key matches panel id
    raw_keys = [k for k in keys if k.startswith("raw_")]
    assert raw_keys == [f"raw_{panel['id']}"]


def test_view_raw_payload_round_trips(gray_2d):
    posts = []

    def fake_post(url, files=None, **kw):
        meta = _parse_meta(files)
        pid = meta["panels"][0]["id"]
        raw_bytes = files[f"raw_{pid}"][1]
        decoded = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        posts.append(decoded)
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_2d, label="x")

    np.testing.assert_array_equal(posts[0], gray_2d)


def test_view_passes_wavelength(gray_2d):
    posts = []

    def fake_post(url, files=None, **kw):
        posts.append(_parse_meta(files))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_2d, kind="mono", wavelength=656.3)

    panel = posts[0]["panels"][0]
    assert panel["kind"] == "mono"
    assert panel["wavelength"] == 656.3


def test_view_multi_with_wavelengths():
    arr = np.random.default_rng(0).random((20, 20, 4), dtype=np.float32)
    posts = []

    def fake_post(url, files=None, **kw):
        posts.append(_parse_meta(files))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(arr, kind="multi", wavelengths=[450, 550, 650, 750])

    panel = posts[0]["panels"][0]
    assert panel["kind"] == "multi"
    assert panel["wavelengths"] == [450, 550, 650, 750]


def test_compare_two_arrays(gray_2d):
    posts = []

    def fake_post(url, files=None, **kw):
        posts.append((_parse_meta(files), set(files.keys())))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.compare(gray_2d, gray_2d + 0.1, label="cmp")

    meta, keys = posts[0]
    assert meta["mode"] == "compare"
    assert len(meta["panels"]) == 2
    raw_keys = [k for k in keys if k.startswith("raw_")]
    assert len(raw_keys) == 2
    assert meta["panels"][0]["label"] == "A"
    assert meta["panels"][1]["label"] == "B"


def test_compare_three_with_labels(gray_2d):
    posts = []

    def fake_post(url, files=None, **kw):
        posts.append(_parse_meta(files))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.compare(gray_2d, gray_2d + 0.1, gray_2d + 0.2,
                    labels=("noisy", "denoised", "gt"))

    meta = posts[0]
    assert len(meta["panels"]) == 3
    assert [p["label"] for p in meta["panels"]] == ["noisy", "denoised", "gt"]


def test_compare_requires_2_plus():
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post") as mp:
        api.compare(np.zeros((4, 4)))  # only one
    mp.assert_not_called()


def test_save_calls_snapshot(tmp_path):
    posted = {}

    def fake_post(url, json=None, **kw):
        posted["url"] = url
        posted["body"] = json
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.save(str(tmp_path / "out.dbv"))

    assert posted["url"].endswith("/snapshot")
    assert posted["body"]["path"].endswith("out.dbv")


def test_view_fallback_on_server_down(gray_2d, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_FALLBACK_DIR", tmp_path)
    with patch.object(api, "ensure_server", return_value=False):
        api.view(gray_2d, label="oops")
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 1
