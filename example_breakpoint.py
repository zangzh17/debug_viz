"""Realistic breakpoint workflow example.

Simulates a small image-processing pipeline. At the breakpoint you can poke
arrays via debug_viz from the debug console / pdb prompt.

Run it:
    python example_breakpoint.py

When the breakpoint hits (pdb prompt), try:
    >>> from debug_viz import view, compare
    >>> view(img)
    >>> view(noisy, label="noisy input")
    >>> view(spectrum, kind="multi", wavelengths=[450, 550, 650, 750])
    >>> compare(noisy, denoised, labels=("input", "output"))
    >>> view(np.abs(noisy - denoised), label="residual", scale="log")
    >>> c                 # continue past the breakpoint

The first view() spawns the server on 127.0.0.1:8765 and opens your browser.
Subsequent calls re-use the same tab.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def pipeline():
    rng = np.random.default_rng(42)

    # A few representative intermediate states
    y, x = np.mgrid[0:1500, 0:1500].astype(np.float32)
    img = np.exp(-((x - 750) ** 2 + (y - 750) ** 2) / (2 * 200 ** 2))

    noisy = img + rng.normal(0, 0.08, size=img.shape).astype(np.float32)
    denoised = gaussian_filter(noisy, sigma=2.5).astype(np.float32)

    # Multi-band "sensor"
    spectrum = np.stack([
        np.exp(-((x[:600, :600] - 200) ** 2 + (y[:600, :600] - 200) ** 2) / 8000),
        np.exp(-((x[:600, :600] - 300) ** 2 + (y[:600, :600] - 300) ** 2) / 8000),
        np.exp(-((x[:600, :600] - 400) ** 2 + (y[:600, :600] - 400) ** 2) / 8000),
        np.exp(-((x[:600, :600] - 450) ** 2 + (y[:600, :600] - 450) ** 2) / 8000),
    ], axis=-1).astype(np.float32)

    # Drop in here and inspect any of these from the pdb prompt.
    breakpoint()

    return denoised


if __name__ == "__main__":
    pipeline()
