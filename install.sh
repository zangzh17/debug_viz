#!/usr/bin/env bash
# Install runtime deps for debug_viz into the current Python env.
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
    uv pip install starlette "uvicorn[standard]" pillow httpx python-multipart numpy
else
    python -m pip install starlette "uvicorn[standard]" pillow httpx python-multipart numpy
fi

echo "debug_viz deps installed."
