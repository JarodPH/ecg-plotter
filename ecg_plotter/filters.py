"""Signal conditioning filters for ECG traces.

Each function operates on a single 1-D lead. Use :func:`apply_to_leads` to
run the same filter across a full 12-lead recording.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from numpy import asarray, ndarray
from scipy.signal import butter, filtfilt, iirnotch, medfilt

from ecg_plotter.leads import LeadInput, _is_dataframe, as_lead_dict

if TYPE_CHECKING:
    from pandas import DataFrame


def _require_filtfilt_length(x: ndarray, b: ndarray, a: ndarray, *, name: str) -> None:
    """Raise a clear error before filtfilt's cryptic padlen failure."""
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        raise ValueError(
            f"{name}: signal length ({len(x)}) is too short for zero-phase filtering "
            f"(needs more than {padlen} samples for this filter). "
            "Use a longer recording, lower filter order, or skip filtering."
        )


def bandpass(
    signal: ndarray,
    fs: float,
    lowcut: float = 0.05,
    highcut: float = 150.0,
    order: int = 4,
) -> ndarray:
    """Zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    signal
        1-D ECG samples in millivolts (mV).
    fs
        Sampling frequency in Hz.
    lowcut, highcut
        Pass-band edges in Hz.
    order
        Filter order.
    """
    x = asarray(signal, dtype=float).reshape(-1)
    if lowcut <= 0 or highcut <= lowcut:
        raise ValueError("Require 0 < lowcut < highcut.")
    nyq = 0.5 * fs
    if highcut >= nyq:
        raise ValueError(f"highcut ({highcut}) must be < Nyquist ({nyq}).")
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    _require_filtfilt_length(x, b, a, name="bandpass")
    return filtfilt(b, a, x)


def notch(
    signal: ndarray,
    fs: float,
    freq: float = 60.0,
    quality: float = 30.0,
) -> ndarray:
    """Zero-phase IIR notch filter (mains rejection).

    Parameters
    ----------
    signal
        1-D ECG samples in millivolts (mV).
    fs
        Sampling frequency in Hz.
    freq
        Notch centre frequency in Hz (typically 50 or 60).
    quality
        Quality factor ``Q = freq / bandwidth``. Higher Q → narrower notch.
    """
    x = asarray(signal, dtype=float).reshape(-1)
    nyq = 0.5 * fs
    if not 0 < freq < nyq:
        raise ValueError(f"Notch frequency must be in (0, {nyq}); got {freq}.")
    b, a = iirnotch(freq, quality, fs=fs)
    _require_filtfilt_length(x, b, a, name="notch")
    return filtfilt(b, a, x)


def median_baseline(
    signal: ndarray,
    kernel_size: int = 201,
) -> ndarray:
    """Remove baseline wander by subtracting a median-filtered estimate.

    ``kernel_size`` must be odd. Larger kernels track slower drift.

    Parameters
    ----------
    signal
        1-D ECG samples in millivolts (mV).
    kernel_size
        Odd window length for the median baseline estimate.
    """
    x = asarray(signal, dtype=float).reshape(-1)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if kernel_size > len(x):
        raise ValueError(
            f"kernel_size ({kernel_size}) cannot exceed signal length ({len(x)})."
        )
    baseline = medfilt(x, kernel_size=kernel_size)
    return x - baseline


def apply_to_leads(
    leads: LeadInput,
    filter_fn: Callable[..., ndarray],
    **kwargs: Any,
) -> dict[str, ndarray] | DataFrame:
    """Apply a 1-D filter function to every lead.

    Accepts a lead mapping, ``(12, n)`` / ``(n, 12)`` array, or DataFrame.
    DataFrame inputs are returned as a DataFrame with the same lead columns;
    other inputs return a lead dictionary.

    Example
    -------
    >>> cleaned = apply_to_leads(leads, bandpass, fs=500, lowcut=0.05, highcut=150)
    >>> cleaned = apply_to_leads(cleaned, notch, fs=500, freq=60)
    """
    was_df = _is_dataframe(leads)
    lead_map = as_lead_dict(leads)
    out = {name: filter_fn(sig, **kwargs) for name, sig in lead_map.items()}
    if was_df:
        from pandas import DataFrame

        return DataFrame(out)
    return out
