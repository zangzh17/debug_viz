from unittest.mock import MagicMock, patch

from debug_viz import boot


def test_ping_failure_then_spawn_then_success(tmp_path, monkeypatch):
    """ensure_server spawns when no server is up, then polls."""
    monkeypatch.setattr(boot, "_PID_FILE", tmp_path / "pid")
    monkeypatch.setattr(boot, "_BROWSER_OPENED", False, raising=False)

    call_count = {"n": 0}

    def fake_get(url, timeout):
        call_count["n"] += 1
        m = MagicMock()
        m.status_code = 200 if call_count["n"] >= 2 else 0
        if call_count["n"] < 2:
            raise OSError("no server")
        return m

    popen_mock = MagicMock()
    popen_mock.return_value.pid = 12345

    with patch.object(boot.httpx, "get", side_effect=fake_get), \
         patch.object(boot.subprocess, "Popen", popen_mock), \
         patch.object(boot.subprocess, "run") as run_mock:
        ok = boot.ensure_server(timeout_sec=2.0)
        assert ok is True
        assert popen_mock.called

    # browser opened once
    assert run_mock.call_count == 1
    # pid file written
    assert (tmp_path / "pid").read_text() == "12345"


def test_already_running_no_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "_PID_FILE", tmp_path / "pid")
    m = MagicMock(); m.status_code = 200
    with patch.object(boot.httpx, "get", return_value=m), \
         patch.object(boot.subprocess, "Popen") as popen_mock:
        ok = boot.ensure_server()
        assert ok is True
        assert not popen_mock.called


def test_browser_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "_BROWSER_OPENED", False, raising=False)
    monkeypatch.setattr(boot, "_PID_FILE", tmp_path / "pid")

    m = MagicMock(); m.status_code = 200
    with patch.object(boot.httpx, "get", return_value=m), \
         patch.object(boot.subprocess, "Popen") as popen_mock, \
         patch.object(boot.subprocess, "run") as run_mock:
        # Server already up, no spawn -> no browser open
        boot.ensure_server()
        assert run_mock.call_count == 0
        assert not popen_mock.called
