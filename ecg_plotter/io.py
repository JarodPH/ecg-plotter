"""Load ECG recordings from WFDB paths or JSON Lines files.

Lead samples are expected in millivolts (mV). ``load_wfdb`` reads
``record.p_signal``, which WFDB already scales to physical units using the
header gain; a warning is issued if those units are not mV.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from numpy import asarray, ndarray, vstack

from ecg_plotter.leads import LEAD_NAMES, as_lead_dict

_MV_ALIASES = frozenset({"mv", "millivolt", "millivolts"})


def _warn_if_units_not_mv(units: Sequence[Any] | None, record_path: Path) -> None:
    if not units:
        return
    unexpected = [
        str(unit)
        for unit in units
        if str(unit).strip().lower() not in _MV_ALIASES
    ]
    if unexpected:
        warnings.warn(
            f"WFDB record {record_path} has non-mV units {unexpected}; "
            "ecg-plotter expects lead amplitudes in millivolts (mV).",
            UserWarning,
            stacklevel=3,
        )


def leads_to_array(
    leads: Mapping[str, ndarray],
    lead_names: Sequence[str] = LEAD_NAMES,
) -> ndarray:
    """Stack a lead mapping into an array of shape ``(n_leads, n_samples)``.

    Values are treated as millivolts (mV).
    """
    missing = [name for name in lead_names if name not in leads]
    if missing:
        raise KeyError(f"Missing leads: {missing}")
    return vstack([asarray(leads[name], dtype=float) for name in lead_names])


def load_wfdb(
    record_path: str | Path,
    *,
    lead_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a WFDB record into a lead dictionary plus metadata.

    Lead arrays come from ``record.p_signal`` and are expected in millivolts
    (mV). If the header ``units`` are not mV, a :class:`UserWarning` is
    emitted; values are still returned unchanged.

    Parameters
    ----------
    record_path
        Path to the record *without* extension, or with ``.hea`` / ``.dat``.
    lead_names
        Optional override for signal names (must match the number of channels).
        If omitted, names are taken from the WFDB header (``record.sig_name``),
        so any 12-lead WFDB dataset works regardless of channel order
        (e.g. ``aVL`` before or after ``aVF``).

    Returns
    -------
    dict
        ``leads`` (name → array in mV), ``fs``, ``sig_name``, ``record_name``,
        ``n_samples``, ``units``, and the raw ``wfdb.Record`` under ``record``.
    """
    from wfdb import rdrecord

    path = Path(record_path)
    if path.suffix.lower() in {".hea", ".dat", ".mat"}:
        path = path.with_suffix("")

    record = rdrecord(str(path))
    signal = asarray(record.p_signal, dtype=float)
    names = list(record.sig_name) if lead_names is None else list(lead_names)
    units = list(record.units) if getattr(record, "units", None) is not None else None
    _warn_if_units_not_mv(units, path)

    if signal.ndim != 2 or signal.shape[1] != len(names):
        raise ValueError(
            f"Signal shape {signal.shape} does not match {len(names)} lead names."
        )

    leads = {name: signal[:, i].copy() for i, name in enumerate(names)}
    return {
        "leads": leads,
        "fs": float(record.fs),
        "sig_name": names,
        "record_name": record.record_name,
        "n_samples": int(record.sig_len),
        "units": units,
        "record": record,
    }


def load_jsonl(
    path: str | Path,
    *,
    lead_names: Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON Lines file of lead arrays.

    Each line must be a JSON object with:

    - lead arrays in millivolts (mV), e.g. ``"I": [...]``
    - required ``fs`` (sampling frequency in Hz; must be positive)
    - optional metadata such as ``id`` / ``label``

    Non-lead keys are returned alongside ``leads``.
    """
    names = set(lead_names) if lead_names is not None else set(LEAD_NAMES)
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_no}; got {type(row)}.")

            lead_map = {
                key: asarray(val, dtype=float).reshape(-1)
                for key, val in row.items()
                if key in names and not isinstance(val, str)
            }
            if not lead_map:
                raise ValueError(f"No lead arrays found on line {line_no} of {path}.")

            meta = {key: val for key, val in row.items() if key not in lead_map}
            if "fs" not in meta or meta["fs"] is None:
                raise ValueError(
                    f"Missing required field 'fs' on line {line_no} of {path}."
                )
            try:
                fs = float(meta["fs"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid 'fs' on line {line_no} of {path}: {meta['fs']!r}."
                ) from exc
            if fs <= 0:
                raise ValueError(
                    f"'fs' must be positive on line {line_no} of {path}; got {fs}."
                )

            yield {
                "leads": as_lead_dict(lead_map),
                "fs": fs,
                "id": meta.get("id", meta.get("study_id")),
                "label": meta.get("label", meta.get("condition")),
                "meta": meta,
            }
