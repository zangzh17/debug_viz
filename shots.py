"""Headless screenshots of the debug_viz UI in various states."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8765/"
OUT = Path("/tmp/dbv_shots")
OUT.mkdir(exist_ok=True)


async def shot(page, name):
    p = OUT / f"{name}.png"
    await page.screenshot(path=str(p))
    print(f"  -> {p}")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 900})
        page = await ctx.new_page()
        page.on("console", lambda m: print("[console]", m.type, m.text) if m.type == "error" else None)
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_function("document.querySelectorAll('.tile').length >= 7", timeout=8000)
        await shot(page, "01_gallery")

        # Click the 6000x6000 frame (oldest -> bottom of gallery)
        await page.evaluate("""() => {
          const tiles = [...document.querySelectorAll('.tile')];
          const t = tiles.find(x => x.querySelector('.label').textContent.includes('6000'));
          t && t.click();
        }""")
        await page.wait_for_selector(".vp-osd canvas", timeout=8000)
        await page.wait_for_timeout(1500)
        await shot(page, "02_viewer_6k")

        # Zoom in via scroll
        viewer = page.locator(".vp-osd").first
        box = await viewer.bounding_box()
        cx, cy = box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5
        await page.mouse.move(cx, cy)
        for _ in range(5):
            await page.mouse.wheel(0, -200)
            await page.wait_for_timeout(120)
        await page.wait_for_timeout(800)
        await shot(page, "03_zoomed_6k")

        # Switch to multispectral frame
        await page.evaluate("""() => {
          const tiles = [...document.querySelectorAll('.tile')];
          const t = tiles.find(x => x.querySelector('.label').textContent.includes('multispectral'));
          t && t.click();
        }""")
        await page.wait_for_selector(".band-strip-mini .band-item", timeout=8000)
        await page.wait_for_timeout(1500)
        await shot(page, "04_multispectral")

        # Adjust scale to log via UI button (first 'log' button in controls)
        await page.evaluate("""() => {
          const btns = [...document.querySelectorAll('.controls .seg button')];
          const lg = btns.find(b => b.textContent.trim() === 'log');
          lg && lg.click();
        }""")
        await page.wait_for_timeout(1800)
        await shot(page, "05_multispectral_log")

        # 3-way compare
        await page.evaluate("""() => {
          const tiles = [...document.querySelectorAll('.tile')];
          const t = tiles.find(x => x.querySelector('.label').textContent.includes('3-way'));
          t && t.click();
        }""")
        await page.wait_for_function("document.querySelectorAll('.vp-osd').length === 3", timeout=8000)
        await page.wait_for_timeout(1500)
        await shot(page, "06_compare_3way")

        # Zoom into one of the compare viewers — should sync the others
        viewer = page.locator(".vp-osd").first
        box = await viewer.bounding_box()
        cx, cy = box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5
        await page.mouse.move(cx, cy)
        for _ in range(6):
            await page.mouse.wheel(0, -200)
            await page.wait_for_timeout(120)
        await page.wait_for_timeout(800)
        await shot(page, "07_compare_synced_zoom")

        # H-alpha mono tinted
        await page.evaluate("""() => {
          const tiles = [...document.querySelectorAll('.tile')];
          const t = tiles.find(x => x.querySelector('.label').textContent.includes('H-alpha'));
          t && t.click();
        }""")
        await page.wait_for_selector(".vp-osd canvas", timeout=8000)
        await page.wait_for_timeout(1500)
        await shot(page, "08_halpha_tinted")

        # Collapse gallery
        await page.click("#gallery_toggle")
        await page.wait_for_timeout(500)
        await shot(page, "09_gallery_collapsed")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
