"""CLI: python -m debug_viz.viewer path.dbv -- replay a saved session."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .boot import base_url


def _ping() -> bool:
    try:
        return httpx.get(f"{base_url()}/health", timeout=0.2).status_code == 200
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="debug_viz.viewer")
    parser.add_argument("path", help=".dbv snapshot to replay")
    parser.add_argument("--no-open", action="store_true", help="don't auto-open browser")
    args = parser.parse_args(argv)

    p = Path(os.path.expanduser(args.path))
    if not p.exists():
        print(f"[debug_viz] no such file: {p}", file=sys.stderr)
        return 2

    if _ping():
        print("[debug_viz] another debug_viz server is already running on 8765; "
              "stop it before replaying.", file=sys.stderr)
        return 3

    proc = subprocess.Popen(
        [sys.executable, "-m", "debug_viz.server", "--load", str(p)],
        start_new_session=True,
    )

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _ping():
            break
        time.sleep(0.1)
    else:
        print("[debug_viz] server failed to come up", file=sys.stderr)
        proc.terminate()
        return 4

    url = base_url()
    if not args.no_open:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", url], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", url], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print(f"[debug_viz] viewer running at {url}  (Ctrl+C to quit)")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
