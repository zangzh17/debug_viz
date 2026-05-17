# debug_viz

Local debug visualizer for Python pipelines that manipulate large image / array
data (6000×6000+ is a first-class size).

While paused at a breakpoint, call `view(x)` or `compare(a, b, c)` from the
debug console. The server renders each array into a Deep Zoom (DZI) tile
pyramid so the browser pans/zooms at native resolution without ever loading
the whole image. Browser-side controls re-render in real time
(linear/log scale, percentile clipping, wavelength tinting, band re-pick).

## Install

```bash
./install.sh
# or:
pip install starlette "uvicorn[standard]" pillow httpx python-multipart numpy
```

## Use

From a paused debugger console:

```python
from debug_viz import view, compare, save

view(big_psf)
view(narrowband, kind="mono", wavelength=656.3)      # H-alpha tint
view(cube, kind="multi", wavelengths=[480, 560, 655, 865])
view(hdr, scale="log", clip_pct=(0.5, 99.5))

compare(noisy, denoised, ground_truth,
        labels=("noisy", "σ=2", "gt"))               # 2–4 panels, synced zoom
save("~/debug/2026-05-17.dbv")
```

First call spawns the server on `127.0.0.1:8765` and opens your browser.
Subsequent calls reuse the same tab.

## UI

Split layout — gallery on the left (collapsible with `g` or the ≡ button),
main viewer on the right.

The main viewer is an [OpenSeadragon](https://openseadragon.github.io/) tile
viewer with:

- **Minimap** (bottom-right) showing the current zoom region in a red rect.
- **Cursor-anchored wheel zoom** (zoom centers on the mouse).
- **Editable frame title** (top) — persisted server-side; survives switching frames.
- **Editable per-panel sub-titles** (compare mode, one per panel).
- **Control panel** at the bottom — sliders/inputs re-render via the server
  while preserving the current viewport bounds.
- **Compare mode** — 2–4 panels side by side, all panning/zooming together.

## API

```python
view(x, *, label=None,
     kind="auto" | "gray" | "mono" | "rgb" | "raw" | "multi" | "mask",
     bands=None,                   # tuple[int,...] — R/G/B channel picks
     wavelength=None,              # nm; tints mono / gray data
     wavelengths=None,             # list[float]; wavelength per band for multi
     scale="linear" | "log",
     clip_pct=(1.0, 99.0)) -> None

compare(*arrays, label=None, labels=None,
        kind="auto" | "gray" | "mono" | "rgb" | "raw" | "multi",
        bands=None, wavelength=None, wavelengths=None,
        scale="linear" | "log", clip_pct=(1.0, 99.0)) -> None

save(path) -> None
```

## Kind hints

| kind | meaning | render |
|---|---|---|
| `auto` | infer from shape/dtype (default) | as below |
| `gray` | 2-D grayscale | percentile-stretched grayscale, NaN→red |
| `mono` | 2-D single-wavelength data | grayscale, optionally tinted by `wavelength=` |
| `rgb` | (H,W,3) RGB image | direct |
| `raw` | bayer/raw sensor data | tagged "raw"; renders as grayscale (no demosaic) |
| `multi` | (H,W,N) multispectral | composite from `wavelengths=` (wavelength-weighted) or `bands=` (R/G/B picks) + per-band thumbnail strip |
| `mask` | 2-valued bool/int (auto only) | yellow / black |

## Wavelength → color

Bruton approximation of CIE 1931 visible spectrum (380–780 nm) with edge
falloff. UV / IR (outside visible) are clamped to attenuated violet / dark
red so they're still distinguishable.

- `kind="mono"` + `wavelength=656.3` → grayscale tinted with H-alpha red
- `kind="multi"` + `wavelengths=[450, 550, 650, ...]` → each band is
  normalized then weighted by its wavelength color and summed (peak
  re-normalized to 255 to preserve hue).
- Click any band in the multispectral strip to view just that band in mono
  (server slices the stored cube and re-renders).

## Browser-side re-rendering

Each frame's raw array is stored on disk (as `.npy`). When you adjust any
control — log/linear toggle, percentile sliders, wavelength inputs, band
picks — the browser POSTs to `/rerender/<panel_id>`. The server re-runs the
render pipeline, writes a fresh DZI pyramid, and the viewer hot-swaps the
tile source while keeping the current viewport bounds.

Cost of a re-render is the same as the initial encode (~0.5 s for 2K²,
~2 s for 6K²); sliders are debounced 200 ms.

## Replay

```bash
python -m debug_viz.viewer ~/debug/2026-05-17.dbv
```

`.dbv` snapshots include the raw arrays + thumbs + current DZI, so re-render
keeps working in a replayed session.

## Performance (typical, SSD)

| input | encode + DZI write | post latency (debugger sees) |
|---|---|---|
| 2000×2000 float32 | ~0.2 s | ~0.5 s |
| 6000×6000 float32 | ~1.8 s | ~4.5 s |
| 1500×1500×4 multispectral | ~0.4 s | ~0.6 s |

The raw `.npy` upload over loopback (~1 GB/s) is included.

## Layout

```
debug_viz/
  __init__.py    re-exports view / compare / save
  api.py         public funcs; serializes arrays as .npy, POSTs to server
  encode.py      render_array(): array → PIL.Image + metadata
  dzi.py         DZI pyramid writer + thumbnail bytes
  spectral.py    wavelength → sRGB + tint helpers
  boot.py        lazy server spawn + browser launch
  server.py      Starlette: /event /rerender /dzi /thumb /snapshot
  viewer.py      python -m debug_viz.viewer <path.dbv>
  web/
    index.html   split layout
    app.js       gallery + OpenSeadragon viewers + sync + controls
    styles.css
    openseadragon.min.js, osd-images/, split.min.js  (vendored)
  tests/         pytest suite
```

## Tests

```bash
pytest debug_viz/tests/ -v          # 58 tests
```
