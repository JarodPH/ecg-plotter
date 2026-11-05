"""Plot output tests: pixel size, colour channels, and footer helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ecg_plotter.leads import LEAD_NAMES
from ecg_plotter.sheet import ANONYMOUS_ID, _resolve_footer_id, plot


def _open(path):
    image = Image.open(path)
    image.load()
    return image


def test_colour_png_is_rgb_with_exact_pixel_size(leads, fs, tmp_path):
    width_px, height_px = 640, 400
    dpi = 100
    path = plot(
        leads,
        tmp_path / "colour.png",
        resolution=(width_px, height_px),
        dpi=dpi,
        fs=fs,
        colour=True,
    )
    image = _open(path)
    assert image.mode == "RGB"
    assert image.size == (width_px, height_px)
    arr = np.asarray(image)
    assert arr.ndim == 3
    assert arr.shape == (height_px, width_px, 3)
    # Pink ECG paper: red channel should dominate over blue on average.
    assert arr[:, :, 0].mean() > arr[:, :, 2].mean()
    dpi_info = image.info.get("dpi")
    assert dpi_info is not None
    assert dpi_info[0] == pytest.approx(dpi, abs=0.01)
    assert dpi_info[1] == pytest.approx(dpi, abs=0.01)


def test_greyscale_png_is_single_channel_with_exact_pixel_size(leads, fs, tmp_path):
    width_px, height_px = 480, 300
    dpi = 100
    path = plot(
        leads,
        tmp_path / "grey.png",
        resolution=(width_px, height_px),
        dpi=dpi,
        fs=fs,
        colour=False,
    )
    image = _open(path)
    assert image.mode == "L"
    assert image.size == (width_px, height_px)
    arr = np.asarray(image)
    assert arr.ndim == 2
    assert arr.shape == (height_px, width_px)


def test_inches_and_resolution_produce_matching_dimensions(leads, fs, tmp_path):
    dpi = 150
    inches = (3.2, 2.0)
    expected = (int(round(inches[0] * dpi)), int(round(inches[1] * dpi)))

    from_inches = plot(
        leads,
        tmp_path / "from_inches.png",
        inches=inches,
        dpi=dpi,
        fs=fs,
        colour=True,
    )
    from_res = plot(
        leads,
        tmp_path / "from_res.png",
        resolution=expected,
        dpi=dpi,
        fs=fs,
        colour=True,
    )
    assert _open(from_inches).size == expected
    assert _open(from_res).size == expected


def test_resolve_footer_id_anonymise():
    assert _resolve_footer_id("40689238", anonymise=True) == ANONYMOUS_ID
    assert _resolve_footer_id(None, anonymise=True) == ANONYMOUS_ID
    assert _resolve_footer_id("40689238", anonymise=False) == "40689238"
    assert _resolve_footer_id(None, anonymise=False) is None


def test_filter_true_changes_image_vs_filter_false(fs, tmp_path):
    """Applying notch/bandpass inside plot must alter the rendered pixels."""
    n = 1000
    t = np.arange(n, dtype=float) / fs
    # Strong 60 Hz contamination so notch+bandpass visibly changes the PNG.
    noisy = {
        name: 0.3 * np.sin(2 * np.pi * 1.5 * t) + 0.8 * np.sin(2 * np.pi * 60.0 * t)
        for name in LEAD_NAMES
    }
    common = dict(
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=False,
        notch_hz=60,
        bandpass_hz=(0.5, 40.0),
    )
    raw = plot(noisy, tmp_path / "raw.png", filter=False, **common)
    filtered = plot(noisy, tmp_path / "filtered.png", filter=True, **common)
    a = np.asarray(_open(raw), dtype=float)
    b = np.asarray(_open(filtered), dtype=float)
    assert a.shape == b.shape
    assert not np.allclose(a, b)
