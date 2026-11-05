"""Lead naming conventions and the classic 3x4 + rhythm layout."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union

from numpy import asarray, ndarray

if TYPE_CHECKING:
    from pandas import DataFrame

# Common published order (aVL before aVF). For WFDB records, prefer the
# header ``sig_name`` via :func:`ecg_plotter.load_wfdb` rather than assuming
# a fixed channel order.
LEAD_NAMES: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

# Classic clinical-style sheet: three rows of four columns, plus Lead II rhythm.
STANDARD_LAYOUT: tuple[tuple[str, ...], ...] = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)

RHYTHM_LEAD = "II"

# Vertical stacking: each lead row occupies this many mV of display space.
ROW_MV = 4.0
N_ROWS = 4  # three lead rows + rhythm
DISPLAY_MV = ROW_MV * N_ROWS  # 16 mV full-scale height

LeadInput = Union[Mapping[str, ndarray], ndarray, "DataFrame"]


def _is_dataframe(obj: Any) -> bool:
    """Return True if ``obj`` is a pandas DataFrame without importing pandas."""
    cls = type(obj)
    return cls.__name__ == "DataFrame" and cls.__module__.startswith("pandas")


def _is_sample_series(value: Any) -> bool:
    """True when a cell looks like a full lead waveform rather than a scalar."""
    if isinstance(value, (str, bytes)):
        return False
    try:
        arr = asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return arr.size > 1


def _resolve_df_lead_columns(
    df: DataFrame,
    lead_names: Sequence[str] | None,
) -> tuple[list[Any], list[str]]:
    """Return ``(columns to read, names to give them)``.

    The two lists are the same when the columns are already lead names.
    An unlabelled 12-column table keeps its own column labels for reading
    and is named with :data:`LEAD_NAMES` in that column order.
    """
    if lead_names is not None:
        missing = [name for name in lead_names if name not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing lead columns: {missing}")
        names = list(lead_names)
        return names, names

    known = list(LEAD_NAMES)
    found = [name for name in known if name in df.columns]
    # Preserve clinical order from LEAD_NAMES where possible.
    ordered = [name for name in LEAD_NAMES if name in found]
    ordered.extend(name for name in found if name not in ordered)
    if ordered:
        return ordered, ordered

    if df.shape[1] == len(LEAD_NAMES):
        # warn -> this function -> _dataframe_to_leads -> as_lead_dict -> caller
        warnings.warn(
            "DataFrame columns are not lead names, so the 12 columns are "
            f"assumed to be {list(LEAD_NAMES)} in that order. "
            "Pass lead_names= if this order is wrong.",
            UserWarning,
            stacklevel=4,
        )
        return list(df.columns), list(LEAD_NAMES)

    raise ValueError(
        "Could not infer lead columns from DataFrame. "
        f"Expected named leads such as {list(LEAD_NAMES)}, or pass lead_names=..."
    )


def _dataframe_to_leads(
    df: DataFrame,
    *,
    lead_names: Sequence[str] | None = None,
) -> dict[str, ndarray]:
    if df.empty:
        raise ValueError("DataFrame is empty.")

    columns, names = _resolve_df_lead_columns(df, lead_names)
    subset = df.loc[:, columns]

    # Index by position so an unlabelled table can be renamed to LEAD_NAMES
    # without looking those names up as column titles.
    if len(subset) == 1 and all(
        _is_sample_series(subset.iloc[0, i]) for i in range(subset.shape[1])
    ):
        # Single-row JSONL-style frame: each lead cell holds an array/list.
        out = {
            name: asarray(subset.iloc[0, i], dtype=float).reshape(-1)
            for i, name in enumerate(names)
        }
    else:
        # Wide time-series frame: rows = samples, columns = leads.
        out = {
            name: asarray(subset.iloc[:, i], dtype=float).reshape(-1)
            for i, name in enumerate(names)
        }

    lengths = {name: len(sig) for name, sig in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All leads must share the same length; got {lengths}.")
    return out


def as_lead_dict(
    leads: LeadInput,
    *,
    lead_names: Sequence[str] | None = None,
) -> dict[str, ndarray]:
    """Normalise lead input to ``{name: 1-D float array}``.

    Amplitudes are expected in millivolts (mV).

    Parameters
    ----------
    leads
        One of:

        - mapping of lead name → samples
        - array of shape ``(12, n)`` or ``(n, 12)``
        - ``pandas.DataFrame`` with lead columns (rows = samples), or a
          single-row frame whose lead cells contain arrays/lists
    lead_names
        Names for array columns/rows, or the DataFrame columns to use.
        Defaults to :data:`LEAD_NAMES` for arrays. For DataFrames, columns
        named as leads are used when this is omitted. A 12-column table with
        no lead names is assumed to be :data:`LEAD_NAMES` in column order,
        and a warning is issued.
    """
    if _is_dataframe(leads):
        return _dataframe_to_leads(leads, lead_names=lead_names)

    if isinstance(leads, Mapping):
        out = {str(k): asarray(v, dtype=float).reshape(-1) for k, v in leads.items()}
        if not out:
            raise ValueError("Lead mapping is empty.")
        lengths = {name: len(sig) for name, sig in out.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All leads must share the same length; got {lengths}.")
        return out

    arr = asarray(leads, dtype=float)
    names = tuple(lead_names) if lead_names is not None else LEAD_NAMES

    if arr.ndim != 2:
        raise ValueError(f"Array input must be 2-D; got shape {arr.shape}.")

    if arr.shape[0] == len(names):
        rows = arr
    elif arr.shape[1] == len(names):
        rows = arr.T
    else:
        raise ValueError(
            f"Expected a dimension of size {len(names)} for leads {list(names)}; "
            f"got shape {arr.shape}."
        )

    return {name: rows[i].reshape(-1) for i, name in enumerate(names)}
