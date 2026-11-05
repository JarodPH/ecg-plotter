"""Tests for paper calibration and size resolution helpers."""

from __future__ import annotations

import pytest

from ecg_plotter.sheet import _resolve_figsize, paper_calibration


def test_paper_calibration_ideal_sheet():
    # 9.84 × 6.3 in @ 300 dpi over 10 s → classic 25 mm/s and 10 mm/mV.
    calib = paper_calibration(9.84, 6.3, dpi=300, duration_s=10.0)
    assert calib["width_px"] == 2952
    assert calib["height_px"] == 1890
    assert calib["speed_mm_s"] == pytest.approx(25.0, abs=0.05)
    assert calib["gain_mm_mv"] == pytest.approx(10.0, abs=0.05)


def test_paper_calibration_rejects_non_positive():
    with pytest.raises(ValueError, match="duration_s"):
        paper_calibration(1.0, 1.0, dpi=100, duration_s=0)
    with pytest.raises(ValueError, match="display_mv"):
        paper_calibration(1.0, 1.0, dpi=100, duration_s=1.0, display_mv=0)
    with pytest.raises(ValueError, match="dpi"):
        paper_calibration(1.0, 1.0, dpi=0, duration_s=1.0)


def test_resolve_figsize_inches_and_resolution():
    assert _resolve_figsize(inches=(9.84, 6.3), resolution=None, dpi=300) == (9.84, 6.3)
    w, h = _resolve_figsize(inches=None, resolution=(2952, 1890), dpi=300)
    assert w == pytest.approx(9.84)
    assert h == pytest.approx(6.3)


def test_resolve_figsize_mutual_exclusion_and_validation():
    with pytest.raises(ValueError, match="only one"):
        _resolve_figsize(inches=(1, 1), resolution=(10, 10), dpi=100)
    with pytest.raises(ValueError, match="Provide"):
        _resolve_figsize(inches=None, resolution=None, dpi=100)
    with pytest.raises(ValueError, match="dpi"):
        _resolve_figsize(inches=(1, 1), resolution=None, dpi=0)
    with pytest.raises(ValueError, match="positive"):
        _resolve_figsize(inches=(-1, 1), resolution=None, dpi=100)
    with pytest.raises(ValueError, match="positive"):
        _resolve_figsize(inches=None, resolution=(0, 100), dpi=100)
