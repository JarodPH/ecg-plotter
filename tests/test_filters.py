"""Tests for signal filters — including that they actually change the signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_plotter.filters import apply_to_leads, bandpass, median_baseline, notch
from ecg_plotter.leads import LEAD_NAMES


def _tone(fs: float, n: int, freq: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n, dtype=float) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def _power_at(signal: np.ndarray, fs: float, freq: float) -> float:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return float(np.abs(spectrum[idx]) ** 2)


def test_notch_attenuates_mains_frequency(fs):
    n = 4000
    signal = _tone(fs, n, 60.0) + 0.2 * _tone(fs, n, 5.0)
    cleaned = notch(signal, fs=fs, freq=60.0)
    assert cleaned.shape == signal.shape
    assert _power_at(cleaned, fs, 60.0) < 0.05 * _power_at(signal, fs, 60.0)
    # Non-notch content should mostly remain.
    assert _power_at(cleaned, fs, 5.0) > 0.5 * _power_at(signal, fs, 5.0)


def test_bandpass_rejects_out_of_band(fs):
    n = 4000
    # 1 Hz (in band) + 180 Hz (above 150 Hz highcut for fs=500)
    signal = _tone(fs, n, 1.0) + _tone(fs, n, 180.0)
    cleaned = bandpass(signal, fs=fs, lowcut=0.5, highcut=40.0)
    assert cleaned.shape == signal.shape
    assert _power_at(cleaned, fs, 180.0) < 0.05 * _power_at(signal, fs, 180.0)
    assert _power_at(cleaned, fs, 1.0) > 0.5 * _power_at(signal, fs, 1.0)


def test_median_baseline_removes_slow_drift(fs):
    n = 2000
    t = np.arange(n, dtype=float) / fs
    drift = 2.0 * np.sin(2 * np.pi * 0.05 * t)
    qrs = 0.5 * np.sin(2 * np.pi * 5.0 * t)
    signal = drift + qrs
    cleaned = median_baseline(signal, kernel_size=201)
    assert cleaned.shape == signal.shape
    # Slow drift energy should drop; faster content should remain.
    assert np.std(cleaned) < np.std(signal)
    assert np.corrcoef(cleaned, qrs)[0, 1] > 0.7


def test_apply_to_leads_dict_and_dataframe(leads, fs):
    as_dict = apply_to_leads(leads, notch, fs=fs, freq=60.0)
    assert isinstance(as_dict, dict)
    assert set(as_dict) == set(LEAD_NAMES)
    assert as_dict["I"].shape == leads["I"].shape

    df = pd.DataFrame(leads)
    as_df = apply_to_leads(df, bandpass, fs=fs, lowcut=0.5, highcut=40.0)
    assert isinstance(as_df, pd.DataFrame)
    assert list(as_df.columns) == list(LEAD_NAMES)
    assert len(as_df) == len(df)


def test_bandpass_invalid_cutoffs(fs):
    x = np.zeros(500)
    with pytest.raises(ValueError, match="lowcut"):
        bandpass(x, fs=fs, lowcut=0, highcut=40)
    with pytest.raises(ValueError, match="Nyquist"):
        bandpass(x, fs=fs, lowcut=0.5, highcut=300)


def test_notch_invalid_frequency(fs):
    x = np.zeros(500)
    with pytest.raises(ValueError, match="Notch frequency"):
        notch(x, fs=fs, freq=0)
    with pytest.raises(ValueError, match="Notch frequency"):
        notch(x, fs=fs, freq=fs)


def test_bandpass_rejects_short_signal(fs):
    x = np.zeros(20)
    with pytest.raises(ValueError, match="too short for zero-phase filtering"):
        bandpass(x, fs=fs, lowcut=0.05, highcut=150.0)


def test_notch_rejects_short_signal(fs):
    x = np.zeros(8)
    with pytest.raises(ValueError, match="too short for zero-phase filtering"):
        notch(x, fs=fs, freq=60.0)


def test_median_baseline_kernel_validation():
    x = np.zeros(100)
    with pytest.raises(ValueError, match="odd"):
        median_baseline(x, kernel_size=10)
    with pytest.raises(ValueError, match="cannot exceed"):
        median_baseline(x, kernel_size=201)
