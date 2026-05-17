import json
from unittest.mock import patch

import numpy as np

from debug_viz import api


def test_sanitize_nan():
    out = api._sanitize_for_json({"a": float("nan"), "b": [1.0, float("inf"), 2.0]})
    assert out["a"] is None
    assert out["b"] == [1.0, None, 2.0]


def test_view_sends_multipart(gray_2d):
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append((url, json.loads(files["metadata"][1]), set(files.keys())))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_2d, label="foo")

    assert len(posts) == 1
    url, meta, keys = posts[0]
    assert url.endswith("/event")
    assert meta["mode"] == "single"
    assert meta["kind"] == "gray"
    assert meta["fmt"] == "jpeg"
    assert meta["label"] == "foo"
    assert keys == {"metadata", "main_full", "main_thumb"}
    assert "NaN" not in json.dumps(meta)


def test_view_with_nan_sanitized(gray_with_nan):
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append(json.loads(files["metadata"][1]))
        return type("R", (), {"status_code": 200})()
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_with_nan)
    assert "NaN" not in json.dumps(posts[0])
    assert posts[0]["stats"]["nan_pct"] is not None


def test_view_multi_sends_band_parts():
    arr = np.random.default_rng(0).random((20, 20, 4), dtype=np.float32)
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append((json.loads(files["metadata"][1]), set(files.keys())))
        return type("R", (), {"status_code": 200})()
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(arr, label="ms")

    meta, keys = posts[0]
    assert meta["kind"] == "multi"
    assert meta["n_bands"] == 4
    assert len(meta["band_meta"]) == 4
    expected = {"metadata", "main_full", "main_thumb"}
    for i in range(4):
        expected.add(f"band_{i}_full")
        expected.add(f"band_{i}_thumb")
    assert keys == expected


def test_view_explicit_kind_raw(gray_2d):
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append(json.loads(files["metadata"][1]))
        return type("R", (), {"status_code": 200})()
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_2d, kind="raw")
    assert posts[0]["kind"] == "raw"
    assert "demosaic" in (posts[0]["note"] or "")


def test_view_bands_selection():
    arr = np.random.default_rng(0).random((10, 10, 5), dtype=np.float32)
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append(json.loads(files["metadata"][1]))
        return type("R", (), {"status_code": 200})()
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(arr, kind="multi", bands=(0, 2, 4))
    assert posts[0]["bands_selected"] == [0, 2, 4]


def test_compare_sends_4_assets(gray_2d):
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append(set(files.keys()))
        return type("R", (), {"status_code": 200})()
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.compare(gray_2d, gray_2d + 0.1, label="cmp")
    assert posts[0] == {"metadata", "full_a", "thumb_a", "full_b", "thumb_b"}


def test_save_calls_snapshot(tmp_path):
    posted = {}
    def fake_post(url, json=None, **kw):
        posted["url"] = url; posted["body"] = json
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
    files = list(tmp_path.glob("*.jpeg"))
    assert len(files) == 1
