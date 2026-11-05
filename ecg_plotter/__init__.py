"""ECG Plotter — filter and render 12-lead ECG traces for research / visualisation.

Not intended for diagnostic use. Dependencies such as matplotlib, scipy, wfdb,
pandas, and Pillow are imported lazily when needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ecg_plotter._version import __version__

__all__ = [
    "ANONYMOUS_ID",
    "LEAD_NAMES",
    "STANDARD_LAYOUT",
    "__version__",
    "apply_to_leads",
    "as_lead_dict",
    "bandpass",
    "leads_to_array",
    "load_jsonl",
    "load_wfdb",
    "median_baseline",
    "notch",
    "paper_calibration",
    "plot",
]

# name -> (module, attribute)
# Drawing lives in ``sheet``, not a module named ``plot``. A submodule called
# ``plot`` would replace the public ``ecg.plot`` function as soon as that
# submodule was imported.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ANONYMOUS_ID": ("ecg_plotter.sheet", "ANONYMOUS_ID"),
    "LEAD_NAMES": ("ecg_plotter.leads", "LEAD_NAMES"),
    "STANDARD_LAYOUT": ("ecg_plotter.leads", "STANDARD_LAYOUT"),
    "apply_to_leads": ("ecg_plotter.filters", "apply_to_leads"),
    "as_lead_dict": ("ecg_plotter.leads", "as_lead_dict"),
    "bandpass": ("ecg_plotter.filters", "bandpass"),
    "leads_to_array": ("ecg_plotter.io", "leads_to_array"),
    "load_jsonl": ("ecg_plotter.io", "load_jsonl"),
    "load_wfdb": ("ecg_plotter.io", "load_wfdb"),
    "median_baseline": ("ecg_plotter.filters", "median_baseline"),
    "notch": ("ecg_plotter.filters", "notch"),
    "paper_calibration": ("ecg_plotter.sheet", "paper_calibration"),
    "plot": ("ecg_plotter.sheet", "plot"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
