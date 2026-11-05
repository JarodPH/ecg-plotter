"""Error-path tests for missing / invalid ECG inputs to plot()."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_plotter.leads import LEAD_NAMES
from ecg_plotter.sheet import plot


def test_missing_layout_lead_raises(leads, fs, tmp_path):
    incomplete = {k: v for k, v in leads.items() if k != "V3"}
    with pytest.raises(KeyError, match="V3"):
        plot(
            incomplete,
            tmp_path / "missing.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )


def test_missing_rhythm_lead_raises(leads, fs, tmp_path):
    incomplete = {k: v for k, v in leads.items() if k != "II"}
    with pytest.raises(KeyError, match="Rhythm lead"):
        plot(
            incomplete,
            tmp_path / "missing_rhythm.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
            # Layout avoids II so the failure is the rhythm strip specifically.
            layout=(
                ("I", "aVR", "V1", "V4"),
                ("III", "aVL", "V2", "V5"),
                ("aVF", "V3", "V6", "I"),
            ),
            rhythm_lead="II",
        )


def test_unequal_lead_lengths_raises_before_plot(leads, fs, tmp_path):
    bad = dict(leads)
    bad["aVL"] = leads["aVL"][: len(leads["aVL"]) // 2]
    with pytest.raises(ValueError, match="same length"):
        plot(
            bad,
            tmp_path / "unequal.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )


def test_remainder_samples_are_dropped_with_a_warning(fs, tmp_path):
    # 1001 is not divisible by 4, so the last sample cannot fill a column.
    n = 1001
    leads = {name: np.zeros(n, dtype=float) for name in LEAD_NAMES}
    with pytest.warns(UserWarning, match="Dropped the last 1 sample"):
        path = plot(
            leads,
            tmp_path / "trim.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )
    assert path.is_file()


def test_recording_too_short_raises(fs, tmp_path):
    short = {name: np.zeros(3, dtype=float) for name in LEAD_NAMES}
    with pytest.raises(ValueError, match="too short"):
        plot(
            short,
            tmp_path / "short.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )


def test_filter_too_short_for_filtfilt_raises(fs, tmp_path):
    # Layout allows n >= 4, but default bandpass needs more samples for filtfilt.
    short = {name: np.zeros(20, dtype=float) for name in LEAD_NAMES}
    with pytest.raises(ValueError, match="too short for zero-phase filtering"):
        plot(
            short,
            tmp_path / "filter_short.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
            filter=True,
            notch_hz=60,
            bandpass_hz=(0.05, 150),
        )


def test_empty_leads_raises(fs, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plot(
            {},
            tmp_path / "empty.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )


def test_invalid_fs_raises(leads, tmp_path):
    with pytest.raises(ValueError, match="fs"):
        plot(
            leads,
            tmp_path / "bad_fs.png",
            resolution=(320, 200),
            dpi=100,
            fs=0,
        )


def test_inches_and_resolution_together_raises(leads, fs, tmp_path):
    with pytest.raises(ValueError, match="only one"):
        plot(
            leads,
            tmp_path / "both.png",
            inches=(3.2, 2.0),
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )


def test_layout_row_wrong_width_raises(leads, fs, tmp_path):
    with pytest.raises(ValueError, match="exactly 4"):
        plot(
            leads,
            tmp_path / "bad_layout.png",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
            layout=(("I", "II", "III"), ("aVR", "aVL", "aVF"), ("V1", "V2", "V3")),
        )
