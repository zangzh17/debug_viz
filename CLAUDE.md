# debug_viz — guidance for Claude Code sessions

Local visualizer that the user calls from a paused Python debugger (`pdb` /
IDE breakpoint console). `view(x)` / `compare(a, b, c)` push arrays to a
local server which renders DZI tile pyramids served to an OpenSeadragon
viewer in the browser.

The README covers public API + features. This file is for things that are
**not obvious from the code** but are load-bearing for working in this repo.

## Who is the user

- Scientific imaging: works with 5000×6000+ float32 arrays, multispectral
  cubes, real wavelength data. "Too small" benchmarks (e.g. 2000²) won't
  exercise the design.
- Comfortable with architecture-level requests and major rewrites. Picks the
  rigorous option when given choices (OpenSeadragon over panzoom, raw-data
  store + server re-render over client-side WebGL, etc.).
- Speaks Chinese in conversation; code + UI labels in English are fine.
- Reads screenshots. After UI changes, run `shots.py` (or `e2e.py`) and
  inspect the PNGs with the Read tool — don't claim "it works" off a green
  test run alone.

## How they want you to work

- **For multi-axis design requests, surface choices via `AskUserQuestion`
  before coding.** Worked great when redesigning the viewer; the user picked
  the rigorous option in every case and let me proceed. Don't guess.
- **Real visual evidence beats unit-test reports.** `pytest` is the floor.
  After non-trivial UI work, also run `e2e.py` (Playwright-driven) and
  open the screenshots in `/tmp/dbv_shots/`.
- **Precision over convenience.** They asked to replace clip-percentile
  sliders with number inputs because sliders aren't precise enough.
  Default to precise inputs for any numeric control.
- **Minimal chrome.** They removed the OSD top-left +/−/home buttons; rely
  on gestures (scroll = zoom @ cursor, dblclick = home).
- **Don't reinvent.** When asked for "log scale dynamic-range truncation,"
  the right move is one number input (`log_dr`) that caps the displayed
  decades to `hi / 10^N`, not a UI for separate min/max log clips.

## Architecture (manifest v3, May 2026)

Pipeline:

```
view(arr)  ──np.save bytes──▶  POST /event ──┐
                                              ├──▶ server stores raw/<panel_id>.npy
                                              │
                                              ├──▶ render_array() → PIL.Image
                                              │
                                              ├──▶ dzi.write_dzi() →
                                              │      dzi/<panel_id>/<render_id>{.dzi,_files/L/x_y.ext}
                                              │
                                              ├──▶ thumbnail_bytes() → thumb/<panel_id>.<ext>
                                              │
                                              └──▶ WS broadcast frame_added
```

Per-control rerender:

```
slider change  ──▶  POST /rerender/<panel_id> {scale, clip_pct, log_dr, bands, wavelength(s), kind}
                          │
                          └──▶ load raw/<panel_id>.npy from disk
                          └──▶ render_array() with new params
                          └──▶ new DZI under fresh render_id
                          └──▶ WS broadcast panel_rerendered
                          │
client receives panel_rerendered  ──▶  viewer.open(new_dzi_url)
                                       (saves viewport bounds, restores on open)
```

Frame schema:
```
{
  id, ts, label, mode: "single"|"compare",
  panels: [
    { id, label, kind, shape, dtype, stats,
      scale, clip_pct, log_dr, bands, wavelength, wavelengths,
      dzi: {width,height,fmt,max_level,...},
      render_id, fmt, n_bands, bands_meta?: [...] }
  ]
}
```

## Things that look weird but are intentional

- **Old DZI render directories are not garbage-collected.** OpenSeadragon's
  navigator (minimap) can still be loading tiles from the previous
  `render_id` when a re-render swaps in a new one. Deleting the old dir
  produces 404s in the console. They're only swept when the frame itself is
  deleted. Don't reinstate `gc_old_renders`.
- **Server adds frame to archive AFTER encoding, not before.** Tried both;
  the "add early, mark pending" path complicated the front-end render code.
  The clean fix for the WS race is `resyncFrames()` on `ws.onopen` — it
  re-fetches `/frames` to recover any broadcast missed during the handshake
  window. **Don't add server-side queueing.**
- **6000² float32 raw → ~144 MB on disk per frame.** Accumulates fast.
  `/tmp/debug_viz/session_<pid>/` is the holding area. Killing the server
  doesn't clean it; that's by design (replayable). Manual cleanup if needed.
- **Threadpool has `max_workers=2`.** Bigger numbers don't help: encoding is
  GIL-bound (numpy + PIL release the GIL, but DZI tile write is mostly IO+PIL
  which contends). Concurrent rerenders queue.
- **`gc_old_renders` is implemented but unused.** Kept for future use if we
  ever want a "clean stale renders" admin endpoint.

## Surfaces to be careful with

- `web/openseadragon.min.js` and `web/osd-images/*` are vendored
  OpenSeadragon 4.1. Don't edit; replace whole-cloth from upstream if
  upgrading.
- `web/split.min.js` is vendored Split.js 1.6.5.
- `MANIFEST_VERSION` bump = old `.dbv` files unreadable. Keep a single
  current format. Currently v3.
- `wireSync` in `app.js` uses event payload (`e.zoom`, `e.center`,
  `e.refPoint`) — NOT viewport polling. Earlier polling implementation
  was racy under wheel-zoom on the master viewer.
- `_normalize_with_clip` constant-array branch: maps `hi <= 0` → black,
  `hi > 0` → white. Don't change without re-checking multispectral
  composites (dead bands would pollute the composite if mapped to white).

## Running things

```bash
# Install (uv-aware)
./install.sh

# Unit tests — must always pass before commit (58 tests, <1s)
pytest debug_viz/tests/ -q

# End-to-end browser tests (41 tests, ~30s) — Playwright-driven
# Spawns its own server on :8770, kills it on exit
python e2e.py

# Headless screenshots — server must already be running with demo data
python demo.py     # pushes 7 frames to current server (or spawns one)
python shots.py    # writes 9 PNGs to /tmp/dbv_shots/

# Manual try-it
python example_breakpoint.py   # hits pdb; type `view(img)` etc.

# Replay a saved session
python -m debug_viz.viewer ~/some.dbv
```

When running the server manually for inspection (not from pdb):
```bash
python -m debug_viz.server --port 8765
```

## Things they've explicitly rejected

- **Sliders for numeric values** — they want number input precision.
- **Built-in OSD buttons** — too visually busy; gestures only.
- **`view(img)` followed by mutating `img`** — they're aware that the
  server snapshots at call time. Don't add live-update mode unless asked.

## Common follow-up directions

If the user asks for something in this rough area, here's the established
shape of the answer:

- "Add a new render param" → thread it through `encode.render_array`
  signature → `api.view`/`compare` kwargs → `server._render_and_store`
  passes it from panel dict → `/rerender` accepts it in the allowed-update
  list → `app.js renderControls()` adds a number-input control. Add
  unit test in `test_encode.py` and e2e assertion.
- "New visualization kind" → extend `KINDS` tuple + `_resolve_kind`
  + a branch in `render_array`. If lossless (mask/NaN-style), set
  `lossless=True` so DZI uses PNG.
- "Live update without breakpoint" → not currently supported; would need
  a polling/streaming API. Ask first.
- "Bigger arrays" → check `_PCT_SAMPLE_LIMIT` in `encode.py` (currently 1M
  samples for percentile estimate; fine up to ~10 GB). DZI generation is
  the bottleneck at very large sizes; could parallelize across levels.

## Memory

Project-level memory lives at
`~/.claude/projects/-home-zihan-Workspace-debug-viz/memory/`. Notable
entries: user profile, feedback re asking before big rewrites, WS-race
lessons. Append there for durable cross-session knowledge; this file is
the public/committed counterpart.
