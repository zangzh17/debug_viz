import json
import math
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
        # capture metadata
        meta_json = files["metadata"][1]
        posts.append(json.loads(meta_json))
        return type("R", (), {"status_code": 200})()

    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.view(gray_2d, label="foo")

    assert len(posts) == 1
    meta = posts[0]
    assert meta["mode"] == "single"
    assert meta["label"] == "foo"
    assert isinstance(meta["shape"], list)
    # no NaN should leak
    flat = json.dumps(meta)
    assert "NaN" not in flat


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


def test_compare_sends_4_pngs(gray_2d):
    posts = []
    def fake_post(url, files=None, **kw):
        posts.append(files)
        return type("R", (), {"status_code": 200})()
    with patch.object(api, "ensure_server", return_value=True), \
         patch.object(api.httpx, "post", side_effect=fake_post):
        api.compare(gray_2d, gray_2d + 0.1, label="cmp")
    files = posts[0]
    assert set(files.keys()) == {"metadata", "full_a", "thumb_a", "full_b", "thumb_b"}
    meta = json.loads(files["metadata"][1])
    assert meta["mode"] == "compare"


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
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 1
