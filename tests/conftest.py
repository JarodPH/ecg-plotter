"""Shared fixtures for ECG plotter tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_plotter.leads import LEAD_NAMES


@pytest.fixture
def fs() -> float:
    return 500.0


@pytest.fixture
def n_samples() -> int:
    # Long enough for 3×4 layout and filtfilt padding; divisible by 4.
    return 1000


@pytest.fixture
def leads(fs: float, n_samples: int) -> dict[str, np.ndarray]:
    """Synthetic 12-lead dict with distinct low-amplitude sinusoids."""
    t = np.arange(n_samples, dtype=float) / fs
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(LEAD_NAMES):
        # Keep amplitudes small (~0.5 mV) so stacked rows don't clip visually.
        out[name] = 0.4 * np.sin(2 * np.pi * (1.2 + 0.05 * i) * t + 0.1 * i)
    return out


@pytest.fixture
def leads_array(leads: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([leads[name] for name in LEAD_NAMES])


@pytest.fixture
def leads_df(leads: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(leads)
