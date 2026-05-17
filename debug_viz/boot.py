"""Lazy server spawn + browser launch."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

_HOST = "127.0.0.1"
_PORT = 8765
_PID_FILE = Path("/tmp/debug_viz/server.pid")
_BROWSER_OPENED = False


def base_url() -> str:
    return f"http://{_HOST}:{_PORT}"


def _ping(timeout: float = 0.2) -> bool:
    try:
        r = httpx.get(f"{base_url()}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _open_browser_once() -> None:
    global _BROWSER_OPENED
    if _BROWSER_OPENED:
        return
    _BROWSER_OPENED = True
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", base_url()], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", base_url()], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            os.startfile(base_url())  # type: ignore[attr-defined]
    except Exception:
        pass


def ensure_server(timeout_sec: float = 5.0) -> bool:
    if _ping(timeout=0.2):
        return True

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "debug_viz.server"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _PID_FILE.write_text(str(proc.pid))
    except Exception:
        return False

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _ping(timeout=0.2):
            _open_browser_once()
            return True
        time.sleep(0.05)
    return False
