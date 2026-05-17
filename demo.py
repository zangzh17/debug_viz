"""debug_viz demo: 6000×6000 float, wavelength-tinted, multispectral, 3-way compare."""
import time
import numpy as np
from debug_viz import view, compare, save

rng = np.random.default_rng(42)

# 1. 6000×6000 float32 — the headline benchmark size
print("[demo] generating 6000×6000 float32 ...")
y, x = np.mgrid[0:6000, 0:6000].astype(np.float32)
psf = np.exp(-((x - 3000) ** 2 + (y - 3000) ** 2) / (2 * 800.0 ** 2))
hdr_field = psf * 1e4 + rng.exponential(1.0, size=psf.shape).astype(np.float32)
t0 = time.time()
view(hdr_field, label="6000×6000 PSF + Poisson noise", scale="log")
print(f"  posted in {time.time() - t0:.2f}s")

# 2. Mono with H-alpha wavelength tint (656.3 nm)
print("[demo] H-alpha tinted mono ...")
ha = np.exp(-((x[:2000, :2000] - 1000) ** 2 + (y[:2000, :2000] - 1000) ** 2) / (2 * 250 ** 2))
view(ha.astype(np.float32), label="H-alpha 2000×2000",
     kind="mono", wavelength=656.3)

# 3. Multispectral with real-ish wavelengths (Landsat 8 OLI-like)
print("[demo] multispectral 4-band 1500×1500 ...")
H = W = 1500
yy, xx = np.mgrid[0:H, 0:W]
ms = np.stack([
    np.exp(-((xx - 400) ** 2 + (yy - 400) ** 2) / 20000),
    np.exp(-((xx - 700) ** 2 + (yy - 700) ** 2) / 25000),
    np.exp(-((xx - 1000) ** 2 + (yy - 1000) ** 2) / 25000),
    np.exp(-((xx - 1100) ** 2 + (yy - 1100) ** 2) / 30000),
], axis=-1).astype(np.float32)
view(ms, label="multispectral B/G/R/NIR",
     kind="multi", wavelengths=[480, 560, 655, 865])

# 4. 2-way compare (classic before/after)
print("[demo] 2-way compare ...")
clean = np.exp(-((x[:1024, :1024] - 512) ** 2 + (y[:1024, :1024] - 512) ** 2) / 30000)
clean = clean.astype(np.float32)
noisy = clean + rng.normal(0, 0.06, size=clean.shape).astype(np.float32)
compare(noisy, clean, labels=("noisy", "clean"), label="2-way denoise eval")

# 5. 3-way compare (variants)
print("[demo] 3-way compare ...")
from scipy.ndimage import gaussian_filter
denoised_g = gaussian_filter(noisy, sigma=2.0)
denoised_h = gaussian_filter(noisy, sigma=4.0)
compare(noisy, denoised_g, denoised_h,
        labels=("noisy", "σ=2.0", "σ=4.0"),
        label="3-way denoise σ-sweep")

# 6. RGB sanity
print("[demo] RGB ...")
rgb = np.zeros((600, 800, 3), dtype=np.uint8)
rgb[..., 0] = np.linspace(0, 255, 800, dtype=np.uint8)[None, :]
rgb[..., 1] = np.linspace(0, 255, 600, dtype=np.uint8)[:, None]
rgb[..., 2] = 100
view(rgb, label="RGB gradient")

# 7. NaN handling
print("[demo] NaN overlay ...")
nan_field = clean.copy()
nan_field[200:280, 400:600] = np.nan
view(nan_field, label="NaN patch (red overlay)")

save("/tmp/debug_viz_demo.dbv")
print("[demo] done. open http://127.0.0.1:8765")
