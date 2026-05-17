# debug_viz

Local debug visualizer for Python pipelines that manipulate large image/array data.

While paused at a breakpoint, call `view(x)` or `compare(a, b)` from the
debug console to render thumbnails into a persistent browser gallery.
Click any tile to inspect at full resolution with zoom + pan. Edit labels
inline, delete frames, save the whole session to a `.dbv` snapshot, and
reopen it later with `python -m debug_viz.viewer path.dbv`.

## Install

Drop the `debug_viz/` directory anywhere your debugger's Python can import from
(e.g. on `sys.path`, or alongside your project). Install runtime deps:

```bash
./install.sh
# or:
pip install starlette "uvicorn[standard]" pillow httpx python-multipart numpy
```

The host project is not touched. Add `debug_viz/` to your project's
`.gitignore` if you don't want to commit it.

## Use

Paused at a breakpoint, in the debug console:

```python
from debug_viz import view, compare, save
view(x)
view(big_img, label="psf", scale="log")
compare(a, b, label="before / after")
save("~/debug/2026-05-16-run.dbv")
```

First call spawns a local server on `127.0.0.1:8765` and opens your browser.
Subsequent calls reuse the same tab.

## Replay

```bash
python -m debug_viz.viewer ~/debug/2026-05-16-run.dbv
```

## API

```python
view(x, *, label=None, scale="linear" | "log", clip_pct=(1.0, 99.0)) -> None
compare(a, b, *, label=None, scale="linear" | "log", clip_pct=(1.0, 99.0)) -> None
save(path) -> None
```

Supported inputs (anything numpy-array-like):

| input | rendering |
|---|---|
| 2-D float | grayscale, 1–99% percentile auto-stretch, NaN → red |
| 2-D bool / 2-valued integer | yellow / black mask |
| 3-D (H,W,3) float in [0,1] | RGB |
| 3-D (H,W,3) uint8/uint16 | normalize per dtype |
| 3-D (H,W,N>3) | first 3 channels + note |
| anything else | degrade fallback (placeholder + note) |

## Tests

```bash
pytest debug_viz/tests/ -v
# in a project with a coverage gate:
pytest debug_viz/tests/ --no-cov -v
```

## Layout

```
debug_viz/
  __init__.py    re-exports view / compare / save
  api.py         public funcs
  encode.py      array -> PNG bytes + thumb + stats
  boot.py        lazy server spawn + browser launch
  server.py      Starlette + WS + disk-backed archive
  viewer.py      python -m debug_viz.viewer <path.dbv>
  web/           index.html + app.js + styles.css + panzoom.min.js
  tests/         pytest suite
```
