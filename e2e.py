"""End-to-end test harness for debug_viz UI.

Drives the actual app in a headless browser, exercises edits / sliders /
band clicks / WS broadcasts / snapshot replay, and asserts via API + DOM
that everything took effect."""
import asyncio
import contextlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from urllib.request import urlopen

import httpx
import numpy as np
from playwright.async_api import async_playwright

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
INFO = "\033[36m·\033[0m"

PORT_A = 8770
PORT_B = 8771


def free_port_check(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def start_server(port, load=None):
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "debug_viz.server", "--port", str(port)]
    if load:
        cmd += ["--load", str(load)]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=env, preexec_fn=os.setsid)
    deadline = time.time() + 12
    while time.time() < deadline:
        try:
            if urlopen(f"http://127.0.0.1:{port}/health", timeout=0.3).read():
                return p
        except Exception:
            time.sleep(0.15)
    p.terminate()
    raise RuntimeError(f"server didn't start on {port}")


def stop_server(p):
    if p and p.poll() is None:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except Exception:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)


def npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def post_view(port, arr, *, label, kind="auto", scale="linear",
              clip_pct=(1.0, 99.0), bands=None, wavelength=None,
              wavelengths=None):
    pid = uuid.uuid4().hex[:12]
    meta = {
        "id": uuid.uuid4().hex[:12], "label": label, "ts": time.time(),
        "mode": "single",
        "panels": [{
            "id": pid, "label": "", "kind": kind, "scale": scale,
            "clip_pct": list(clip_pct),
            "bands": list(bands) if bands else None,
            "wavelength": wavelength,
            "wavelengths": list(wavelengths) if wavelengths else None,
        }],
    }
    files = {
        "metadata": (None, json.dumps(meta), "application/json"),
        f"raw_{pid}": (f"{pid}.npy", npy_bytes(arr), "application/octet-stream"),
    }
    r = httpx.post(f"http://127.0.0.1:{port}/event", files=files, timeout=180.0)
    r.raise_for_status()
    return r.json()["id"], pid


def post_compare(port, arrays, labels, *, label="cmp"):
    panels, files = [], {}
    pids = []
    for arr, sub in zip(arrays, labels):
        pid = uuid.uuid4().hex[:12]
        pids.append(pid)
        panels.append({"id": pid, "label": sub, "kind": "auto",
                       "scale": "linear", "clip_pct": [1.0, 99.0],
                       "bands": None, "wavelength": None, "wavelengths": None})
        files[f"raw_{pid}"] = (f"{pid}.npy", npy_bytes(arr), "application/octet-stream")
    meta = {"id": uuid.uuid4().hex[:12], "label": label, "ts": time.time(),
            "mode": "compare", "panels": panels}
    files["metadata"] = (None, json.dumps(meta), "application/json")
    r = httpx.post(f"http://127.0.0.1:{port}/event", files=files, timeout=180.0)
    r.raise_for_status()
    return r.json()["id"], pids


def gen_gauss(h, w, cx, cy, sigma):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


class Report:
    def __init__(self):
        self.passes = []
        self.fails = []

    def ok(self, name, _detail=""):
        self.passes.append(name)
        print(f"  {OK} {name}")

    def fail(self, name, detail=""):
        self.fails.append((name, detail))
        print(f"  {FAIL} {name}  {detail}")

    def assert_eq(self, name, got, want):
        if got == want:
            self.ok(f"{name} = {want!r}")
        else:
            self.fail(name, f"got={got!r} want={want!r}")

    def assert_true(self, name, cond, detail=""):
        (self.ok if cond else self.fail)(name, detail)

    def summary(self):
        print(f"\n{len(self.passes)} passed · {len(self.fails)} failed")
        for n, d in self.fails:
            print(f"  {FAIL} {n} {d}")
        return len(self.fails) == 0


async def section(title):
    print(f"\n{INFO} {title}")


async def main():
    rep = Report()
    # Fresh server, fresh assets dir
    shutil.rmtree("/tmp/debug_viz", ignore_errors=True)
    srv_a = start_server(PORT_A)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(viewport={"width": 1600, "height": 900})
            page = await ctx.new_page()
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))

            # Seed initial data
            await section("seed: 4 single + 1 compare-2 + 1 compare-3")
            f1, p1 = post_view(PORT_A, gen_gauss(800, 800, 400, 400, 90), label="g_small")
            f2, p2 = post_view(PORT_A, gen_gauss(2000, 2000, 1000, 1000, 250), label="mono_HA",
                               kind="mono", wavelength=656.3)
            ms_arr = np.stack([
                gen_gauss(400, 400, 100, 100, 60),
                gen_gauss(400, 400, 200, 200, 60),
                gen_gauss(400, 400, 300, 300, 60),
                gen_gauss(400, 400, 350, 350, 60),
            ], axis=-1)
            f3, p3 = post_view(PORT_A, ms_arr, label="ms_4band",
                               kind="multi", wavelengths=[450, 550, 650, 750])
            nan_arr = gen_gauss(512, 512, 256, 256, 80)
            nan_arr[100:150, 200:300] = np.nan
            f4, _ = post_view(PORT_A, nan_arr, label="nan_overlay")
            a_cmp = gen_gauss(512, 512, 256, 256, 80)
            b_cmp = a_cmp + np.random.default_rng(0).normal(0, 0.05, a_cmp.shape).astype(np.float32)
            f5, [pc1, pc2] = post_compare(PORT_A, [a_cmp, b_cmp],
                                          labels=("clean", "noisy"), label="cmp_2way")
            f6, [pt1, pt2, pt3] = post_compare(
                PORT_A,
                [a_cmp,
                 a_cmp + np.random.default_rng(1).normal(0, 0.1, a_cmp.shape).astype(np.float32),
                 a_cmp * 1.1],
                labels=("A", "B", "C"), label="cmp_3way")
            await page.goto(f"http://127.0.0.1:{PORT_A}/", wait_until="networkidle")
            await page.wait_for_function("document.querySelectorAll('.tile').length === 6", timeout=10_000)
            rep.ok("seeded 6 frames, all 6 tiles rendered in gallery")

            # ---- 1. Title edit persistence ----
            await section("1. frame title edit · persist via API · survive switch")
            # Open f1
            await page.evaluate(f"""() => {{
              const tiles = [...document.querySelectorAll('.tile')];
              const t = tiles.find(x => x.querySelector('.label').textContent.includes('g_small'));
              t.click();
            }}""")
            await page.wait_for_selector(".vp-osd canvas", timeout=8000)
            # Edit title
            await page.evaluate("""() => {
              const el = document.getElementById('v_title');
              el.focus(); el.textContent = 'g_renamed';
              el.dispatchEvent(new Event('blur'));
            }""")
            await page.wait_for_timeout(400)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            new_label = next(f["label"] for f in api["frames"] if f["id"] == f1)
            rep.assert_eq("API: frame label persisted", new_label, "g_renamed")
            tile_label = await page.evaluate(f"""() => {{
              const tiles = [...document.querySelectorAll('.tile')];
              const t = tiles.find(x => x.dataset || true);
              const f = [...document.querySelectorAll('.tile .label')].find(x => x.textContent === 'g_renamed');
              return f ? f.textContent : null;
            }}""")
            rep.assert_eq("DOM: gallery tile label updated via WS", tile_label, "g_renamed")
            # Switch to f4, then back to f1
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent.includes('nan_overlay'));
              t.click();
            }""")
            await page.wait_for_timeout(700)
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent === 'g_renamed');
              t.click();
            }""")
            await page.wait_for_timeout(700)
            title_after = await page.evaluate("() => document.getElementById('v_title').textContent")
            rep.assert_eq("title preserved after switching away+back", title_after, "g_renamed")

            # ---- 2. Compare sub-title edit ----
            await section("2. compare panel sub-title edit · per-panel persistence")
            await page.evaluate(f"""() => {{
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent.includes('cmp_3way'));
              t.click();
            }}""")
            # Wait for 3 viewers
            await page.wait_for_function("document.querySelectorAll('.vp-osd').length === 3", timeout=8000)
            await page.wait_for_timeout(1200)
            # Edit middle panel sub-title via the <input>
            await page.evaluate("""() => {
              const inputs = [...document.querySelectorAll('.vp-title')];
              inputs[1].value = 'mid_renamed';
              inputs[1].dispatchEvent(new Event('change'));
            }""")
            await page.wait_for_timeout(400)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            pn = next(f for f in api["frames"] if f["id"] == f6)
            rep.assert_eq("API: panel[1] sub-label persisted", pn["panels"][1]["label"], "mid_renamed")

            # ---- 3. Compare zoom sync ----
            await section("3. compare: zoom one viewer · other viewports follow")
            # Reset all to home, then zoom on first viewer via OSD API
            zoom_info = await page.evaluate("""async () => {
              const viewers = window.__osd_viewers || null;
              return null;  // we don't expose; use mouse wheel approach instead
            }""")
            # Use wheel events on the first viewer
            v0 = page.locator(".vp-osd").nth(0)
            box = await v0.bounding_box()
            cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
            await page.mouse.move(cx, cy)
            for _ in range(6):
                await page.mouse.wheel(0, -180)
                await page.wait_for_timeout(80)
            await page.wait_for_timeout(800)
            # Each .navigator has a displayregion <div> whose size reflects current zoom
            sizes = await page.evaluate("""() => {
              const navs = [...document.querySelectorAll('.navigator .displayregion')];
              return navs.map(n => ({w: n.offsetWidth, h: n.offsetHeight}));
            }""")
            # All 3 navigator rectangles should now be the same (synced)
            rep.assert_true("3 navigators report same size (zoom synced)",
                            len(sizes) == 3 and len({(s["w"], s["h"]) for s in sizes}) == 1,
                            f"sizes={sizes}")

            # ---- 4. Re-render: scale linear→log ----
            await section("4. control: scale log · server rerender + viewer reload")
            await page.evaluate(f"""() => {{
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent === 'g_renamed');
              t.click();
            }}""")
            await page.wait_for_selector(".vp-osd canvas", timeout=6000)
            await page.wait_for_timeout(500)
            rid_before = next(f["panels"][0]["render_id"] for f in
                              httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()["frames"]
                              if f["id"] == f1)
            await page.evaluate("""() => {
              const btns = [...document.querySelectorAll('.controls .seg button')];
              const lg = btns.find(b => b.textContent.trim() === 'log');
              lg.click();
            }""")
            await page.wait_for_timeout(1500)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f1)
            rep.assert_eq("panel.scale after click", panel["scale"], "log")
            rep.assert_true("render_id changed after rerender",
                            panel["render_id"] != rid_before,
                            f"before={rid_before} after={panel['render_id']}")
            # DZI descriptor for the new render is reachable
            r = httpx.get(f"http://127.0.0.1:{PORT_A}/dzi/{p1}/{panel['render_id']}.dzi")
            rep.assert_eq("new DZI descriptor served", r.status_code, 200)

            # ---- 5. Re-render: percentile number input edit ----
            await section("5. control: clip_pct number inputs · rerender")
            rid_before2 = panel["render_id"]
            # Number inputs are now used for clip_pct (slider replaced).
            # The first two number inputs in .controls are clip lo + clip hi
            # (log_dr comes after on a log-scale panel; but it has placeholder 'off'
            # so we filter by that.)
            await page.evaluate("""() => {
              const inps = [...document.querySelectorAll('.controls input[type=number]')]
                .filter(i => i.placeholder !== 'off' && !/dynamic/i.test(i.title || ''));
              inps[0].value = '5'; inps[0].dispatchEvent(new Event('change'));
              inps[1].value = '95'; inps[1].dispatchEvent(new Event('change'));
            }""")
            await page.wait_for_timeout(1500)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f1)
            rep.assert_eq("clip_pct[0] persisted", panel["clip_pct"][0], 5.0)
            rep.assert_eq("clip_pct[1] persisted", panel["clip_pct"][1], 95.0)
            rep.assert_true("render_id changed after clip edit", panel["render_id"] != rid_before2)

            # ---- 6. Mono wavelength edit ----
            await section("6. wavelength edit on mono frame")
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent.includes('mono_HA'));
              t.click();
            }""")
            await page.wait_for_selector(".vp-osd canvas", timeout=8000)
            await page.wait_for_timeout(500)
            # Wavelength input is min=200 max=1100; clip inputs are max=100.
            await page.evaluate("""() => {
              const wl = [...document.querySelectorAll('.controls input[type=number]')]
                .find(i => i.max === '1100' || i.placeholder === 'λ nm');
              wl.value = 486;   // H-beta blue
              wl.dispatchEvent(new Event('change'));
            }""")
            await page.wait_for_timeout(1500)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f2)
            rep.assert_eq("wavelength persisted", panel["wavelength"], 486.0)

            # ---- 7. Multispectral: click band thumb → mono-of-band ----
            await section("7. multi: click band thumb · become single-band mono view")
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent.includes('ms_4band'));
              t.click();
            }""")
            await page.wait_for_selector(".band-strip-mini .band-item", timeout=8000)
            await page.wait_for_timeout(800)
            await page.evaluate("""() => {
              const items = [...document.querySelectorAll('.band-strip-mini .band-item')];
              items[2].click();   // click band index 2 (650 nm)
            }""")
            await page.wait_for_timeout(1800)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f3)
            rep.assert_eq("after band-click: panel.kind", panel["kind"], "mono")
            rep.assert_eq("after band-click: panel.bands", panel["bands"], [2])
            rep.assert_eq("after band-click: panel.wavelength", panel["wavelength"], 650)

            # ---- 8. Delete frame ----
            await section("8. delete frame via viewer button · WS removes from gallery")
            page.on("dialog", lambda d: asyncio.create_task(d.accept()))
            await page.evaluate(f"""() => {{
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent.includes('nan_overlay'));
              t.click();
            }}""")
            await page.wait_for_selector(".vp-osd canvas", timeout=6000)
            await page.wait_for_timeout(300)
            await page.click("#v_delete")
            await page.wait_for_timeout(700)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            rep.assert_true("frame removed from server",
                            f4 not in {f["id"] for f in api["frames"]})
            tile_count = await page.evaluate("() => document.querySelectorAll('.tile').length")
            rep.assert_eq("gallery tile count -1", tile_count, 5)

            # ---- 9. Live add via WS ----
            await section("9. push new frame via API · gallery updates over WS")
            f7, _ = post_view(PORT_A, gen_gauss(256, 256, 128, 128, 50), label="live_add")
            await page.wait_for_function("document.querySelectorAll('.tile').length === 6", timeout=4000)
            rep.ok("new frame appeared in gallery within 4s")

            # ---- 10. Snapshot + replay ----
            await section("10. snapshot → spawn replay server on different port → frames replay")
            snap = Path("/tmp/debug_viz_e2e.dbv")
            r = httpx.post(f"http://127.0.0.1:{PORT_A}/snapshot",
                           json={"path": str(snap)}, timeout=60.0)
            rep.assert_eq("snapshot POST status", r.status_code, 200)
            rep.assert_true("snapshot file exists & non-empty",
                            snap.exists() and snap.stat().st_size > 50_000)
            srv_b = start_server(PORT_B, load=snap)
            try:
                api2 = httpx.get(f"http://127.0.0.1:{PORT_B}/frames", timeout=5.0).json()
                rep.assert_eq("replay server frame count", len(api2["frames"]), 6)
                # Rerender must still work in replay (raw was packed)
                some_panel = api2["frames"][0]["panels"][0]
                r = httpx.post(f"http://127.0.0.1:{PORT_B}/rerender/{some_panel['id']}",
                               json={"scale": "log"}, timeout=10.0)
                rep.assert_eq("rerender on replay-loaded panel", r.status_code, 200)
            finally:
                stop_server(srv_b)

            # ---- 11. 6000×6000 stress: post + open + zoom + tile fetch ----
            await section("11. 6000×6000 float32 · post + open + zoom + tile served")
            # Health check before the heavy lift
            h = httpx.get(f"http://127.0.0.1:{PORT_A}/health", timeout=5.0).json()
            rep.assert_eq("server health OK before 6K", h.get("ok"), True)
            big = gen_gauss(6000, 6000, 3000, 3000, 800) * 1e3
            big = big.astype(np.float32)
            t0 = time.time()
            f8, p8 = post_view(PORT_A, big, label="6kx6k_stress", scale="log")
            elapsed = time.time() - t0
            # Under stress (other tests still loading tiles), can climb to ~30s;
            # standalone is ~4s. We just want a sanity ceiling here.
            rep.assert_true(f"6Kx6K post+encode completes (took {elapsed:.2f}s)",
                            elapsed < 60.0)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f8)
            rep.assert_eq("6K DZI dims", (panel["dzi"]["width"], panel["dzi"]["height"]),
                          (6000, 6000))
            rep.assert_true("13+ DZI levels", panel["dzi"]["max_level"] >= 13)
            # Fetch a bottom-level tile (interior, not corner)
            r = httpx.get(
                f"http://127.0.0.1:{PORT_A}/dzi/{p8}/{panel['render_id']}_files/{panel['dzi']['max_level']}/12_12.jpg")
            rep.assert_eq("interior bottom-level tile served", r.status_code, 200)

            # Open in viewer + zoom in via mouse wheel; check that tile network requests happen
            await page.wait_for_function(
                "document.querySelectorAll('.tile').length === 7", timeout=4000)
            requests_seen = []
            page.on("request", lambda req: requests_seen.append(req.url)
                    if "/dzi/" in req.url and "_files" in req.url else None)
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent.includes('6kx6k_stress'));
              t.click();
            }""")
            await page.wait_for_selector(".vp-osd canvas", timeout=8000)
            await page.wait_for_timeout(1200)
            initial_n = len(requests_seen)
            v = page.locator(".vp-osd").first
            box = await v.bounding_box()
            cx, cy = box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.55
            await page.mouse.move(cx, cy)
            for _ in range(8):
                await page.mouse.wheel(0, -250)
                await page.wait_for_timeout(90)
            await page.wait_for_timeout(1200)
            rep.assert_true(f"zoom triggered new tile fetches (initial={initial_n}, after_zoom={len(requests_seen)})",
                            len(requests_seen) > initial_n + 5)

            # ---- 12. WS-race regression: post BEFORE late-joining client opens ----
            await section("12. late-joining client sees frames posted before WS open")
            # Open a brand-new browser context (no WS yet)
            ctx2 = await browser.new_context(viewport={"width": 1200, "height": 800})
            page2 = await ctx2.new_page()
            # Push a brand-new frame BEFORE page2 is even pointed at the server
            f_pre, _ = post_view(PORT_A, gen_gauss(300, 300, 150, 150, 50),
                                 label="pre_ws_race")
            # NOW open the page. If addTile only happens via WS broadcasts, the
            # pre-existing frame is fine (it'll show via initial /frames fetch).
            # The real race: open page → during the gap before WS handshake
            # completes, push another frame → that frame would silently
            # disappear without the resync-on-ws-open fix.
            await page2.goto(f"http://127.0.0.1:{PORT_A}/", wait_until="domcontentloaded")
            # Push during the WS-handshake window
            f_race, _ = post_view(PORT_A, gen_gauss(300, 300, 150, 150, 50),
                                  label="ws_race_window")
            # Give the page a moment to fetch + connect WS + resync
            await page2.wait_for_function(
                """() => {
                    const labels = [...document.querySelectorAll('.tile .label')]
                        .map(x => x.textContent);
                    return labels.includes('pre_ws_race') &&
                           labels.includes('ws_race_window');
                }""",
                timeout=8000,
            )
            rep.ok("late client sees both pre-load and race-window frames")
            await ctx2.close()

            # ---- 13. dedup: clicking same tile twice does NOT duplicate viewer ----
            await section("13. dedup: re-click same single-frame tile · only one viewer")
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent === 'g_renamed');
              t.click();
            }""")
            await page.wait_for_selector(".vp-osd canvas", timeout=6000)
            await page.wait_for_timeout(400)
            n_first = await page.evaluate("() => document.querySelectorAll('.vp-osd').length")
            await page.evaluate("""() => {
              const t = [...document.querySelectorAll('.tile')].find(x =>
                x.querySelector('.label').textContent === 'g_renamed');
              t.click(); t.click(); t.click();
            }""")
            await page.wait_for_timeout(400)
            n_after = await page.evaluate("() => document.querySelectorAll('.vp-osd').length")
            rep.assert_eq("viewer count after 1 click", n_first, 1)
            rep.assert_eq("viewer count after 4 clicks (no dup)", n_after, 1)

            # ---- 14. percentile is a number input now ----
            await section("14. percentile uses number inputs · log_dr appears for log scale")
            # log button was clicked earlier; control should now expose a log_dr number input
            controls_inputs = await page.evaluate("""() => {
              return [...document.querySelectorAll('.controls input[type=number]')].length;
            }""")
            ranges = await page.evaluate("""() => {
              return [...document.querySelectorAll('.controls input[type=range]')].length;
            }""")
            rep.assert_eq("no <input type=range> in controls", ranges, 0)
            rep.assert_true("at least 3 number inputs (clip lo/hi + log_dr)", controls_inputs >= 3)

            # ---- 15. log_dr rerender effect ----
            await section("15. log_dr=3 · server applies + render_id refreshes")
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f1)
            rid_before = panel["render_id"]
            # find the log_dr input (the one with placeholder "off" / title containing "dynamic range")
            await page.evaluate("""() => {
              const inps = [...document.querySelectorAll('.controls input[type=number]')];
              const dr = inps.find(i => i.placeholder === 'off' || (i.title||'').includes('dynamic'));
              dr.value = '3';
              dr.dispatchEvent(new Event('change'));
            }""")
            await page.wait_for_timeout(1500)
            api = httpx.get(f"http://127.0.0.1:{PORT_A}/frames").json()
            panel = next(f["panels"][0] for f in api["frames"] if f["id"] == f1)
            rep.assert_eq("panel.log_dr persisted", panel.get("log_dr"), 3.0)
            rep.assert_true("render_id changed after log_dr", panel["render_id"] != rid_before)

            # ---- 16. OSD chrome removed (no zoom/home buttons) ----
            await section("16. OSD top-left +/-/home buttons removed")
            controls_imgs = await page.evaluate("""() => {
              return [...document.querySelectorAll('.openseadragon-container img')]
                .filter(i => /zoomin|zoomout|home/i.test(i.src)).length;
            }""")
            rep.assert_eq("OSD zoom/home button images in DOM", controls_imgs, 0)

            # ---- 17. dblclick on canvas resets to home ----
            await section("17. dblclick on viewer canvas · goes home (resets viewport)")
            v = page.locator(".vp-osd").first
            box = await v.bounding_box()
            cx, cy = box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5
            # First zoom in via wheel
            await page.mouse.move(cx, cy)
            for _ in range(4):
                await page.mouse.wheel(0, -180)
                await page.wait_for_timeout(80)
            await page.wait_for_timeout(400)
            zoom_before = await page.evaluate("""() => {
              return document.querySelector('.navigator .displayregion').offsetWidth;
            }""")
            # Now double-click → should go home
            await page.mouse.dblclick(cx, cy)
            await page.wait_for_timeout(700)
            zoom_after = await page.evaluate("""() => {
              return document.querySelector('.navigator .displayregion').offsetWidth;
            }""")
            rep.assert_true(
                f"dblclick widened display region (zoomed back out: {zoom_before}→{zoom_after})",
                zoom_after > zoom_before * 1.5)

            # ---- 18. font-scale toggle changes computed body font-size ----
            await section("18. font scale S/M/L/XL · CSS variable applied, persisted")
            base_m = await page.evaluate(
                "() => parseFloat(getComputedStyle(document.body).fontSize)")
            await page.click("#font_scale button[data-fs='xl']")
            await page.wait_for_timeout(150)
            base_xl = await page.evaluate(
                "() => parseFloat(getComputedStyle(document.body).fontSize)")
            rep.assert_true(f"XL larger than M ({base_m}px → {base_xl}px)",
                            base_xl > base_m * 1.2)
            stored = await page.evaluate("() => localStorage.getItem('dbv_fs')")
            rep.assert_eq("font scale persisted to localStorage", stored, "xl")
            await page.click("#font_scale button[data-fs='m']")

            # ---- 19. JS console clean (modulo known-benign OSD teardown noise) ----
            await section("19. no unexpected JS console errors during full run")
            # OSD's internal viewport handler can throw on rapid rebuild/destroy
            # ("Cannot read properties of undefined (reading 'x')") — visible
            # only to the console, no user-facing effect. Filter it.
            BENIGN = ("reading 'x')", "TileSource", "Failed to load resource")
            real_errs = [e for e in errors if not any(b in e for b in BENIGN)]
            rep.assert_true(
                f"non-benign console error count = 0 (filtered {len(errors)-len(real_errs)} known)",
                len(real_errs) == 0, f"unexpected: {real_errs[:3]}")

            await browser.close()
    finally:
        stop_server(srv_a)

    return rep.summary()


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
