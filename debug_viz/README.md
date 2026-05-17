# debug_viz

Local debug visualizer for Python pipelines that manipulate large image/array data.

While paused at a breakpoint, call `view(x)` or `compare(a, b)` from the
debug console to render thumbnails into a persistent browser gallery.
Click any tile to inspect at full resolution with zoom + pan. Edit labels
inline, delete frames, save the whole session to a `.dbv` snapshot, and
reopen it later with `python -m debug_viz.viewer path.dbv`.

## Install

Drop the `debug_viz/` directory anywhere your debugger's Python can import from.
Then:

```bash
./install.sh
# or:
pip install starlette "uvicorn[standard]" pillow httpx python-multipart numpy
```

## Use

Paused at a breakpoint, in the debug console:

```python
from debug_viz import view, compare, save

view(x)
view(big_img, label="psf", scale="log")
view(spectral, label="ms data", kind="multi", bands=(0, 4, 7))
compare(a, b, label="before / after")
save("~/debug/2026-05-17-run.dbv")
```

First call spawns a local server on `127.0.0.1:8765` and opens your browser.
Subsequent calls reuse the same tab.

## Replay

```bash
python -m debug_viz.viewer ~/debug/2026-05-17-run.dbv
```

## API

```python
view(x, *, label=None,
     kind="auto" | "gray" | "mono" | "rgb" | "raw" | "multi",
     bands=None,                  # tuple[int,...] for multi composite RGB
     scale="linear" | "log",
     clip_pct=(1.0, 99.0)) -> None

compare(a, b, *, label=None,
        kind="auto" | "gray" | "mono" | "rgb" | "raw",
        scale="linear" | "log",
        clip_pct=(1.0, 99.0)) -> None

save(path) -> None
```

## Kind hints

| kind | meaning | render |
|---|---|---|
| `auto` | infer from shape/dtype (default) | as below |
| `gray` | 2-D grayscale | percentile-stretched grayscale, NaN→red |
| `mono` | 2-D single-wavelength data | same as `gray`, but tagged "mono" in UI |
| `rgb` | (H,W,3) RGB image | direct |
| `raw` | bayer/raw sensor data | tagged "raw"; renders as grayscale (no demosaic) |
| `multi` | (H,W,N) multispectral | composite RGB from `bands` + per-channel strip in modal |
| `mask` | 2-valued bool/int (auto only) | yellow / black |

For `kind="multi"`, the modal shows the composite plus a clickable strip of all
N per-band thumbnails — click any band to swap into the main view at full
resolution.

## Performance

Optimized for 6000×6000-class images:

| input | encode time |
|---|---|
| 6000×6000 float32 | ~1.4s |
| 6000×6000 float32, log scale | ~1.9s |
| 6000×6000 uint16 | ~2.5s |
| 2000×2000×10 multispectral (10 bands) | ~2.6s |

Tricks: numpy-stride downsampled thumbnails (skip full-image resize),
percentile from a 1M-sample subset, float32 (not float64) throughout,
JPEG (lossy q=95) for the full image with automatic PNG fallback when
the render needs exact red (NaN overlay) or yellow (mask) markers.

## Tests

```bash
pytest debug_viz/tests/ -v          # 41 unit tests
pytest debug_viz/tests/ --no-cov -v # in a project with a coverage gate
```

## Layout

```
debug_viz/
  __init__.py    re-exports view / compare / save
  api.py         public funcs
  encode.py      array -> JPEG/PNG bytes + thumb + per-band + stats
  boot.py        lazy server spawn + browser launch
  server.py      Starlette + WS + disk-backed asset archive
  viewer.py      python -m debug_viz.viewer <path.dbv>
  web/           index.html + app.js + styles.css + panzoom.min.js
  tests/         pytest suite
```
