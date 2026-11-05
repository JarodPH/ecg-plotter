"""Render a classic 12-lead ECG sheet to PNG, PDF, or SVG."""

from __future__ import annotations

import io
import warnings
from collections.abc import Sequence
from pathlib import Path

from numpy import arange

from ecg_plotter.leads import (
    DISPLAY_MV,
    RHYTHM_LEAD,
    ROW_MV,
    STANDARD_LAYOUT,
    LeadInput,
    as_lead_dict,
)

# Classic ECG thermal-paper look (RGB).
_PAPER_FACE = "#FFF5F5"
_GRID_MINOR = "#F2C4C4"
_GRID_MAJOR = "#E09090"
_DIVIDER = "#8B3030"
_TRACE = "#1A1A1A"
_LABEL = "#1A1A1A"

# Greyscale sheet.
_GREY_FACE = "white"
_GREY_MINOR = "#B0B0B0"
_GREY_MAJOR = "#696969"
_GREY_DIVIDER = "#2A2A2A"

# Lead-panel separators (heavier than major grid lines).
_DIVIDER_LINEWIDTH = 0.9

# Fixed footer ID used when ``anonymise=True``.
ANONYMOUS_ID = "15030416"


def paper_calibration(
    width_in: float,
    height_in: float,
    *,
    dpi: int,
    duration_s: float,
    display_mv: float = DISPLAY_MV,
) -> dict[str, float]:
    """Compute physical paper speed and gain from figure geometry.

    Pixel rounding is applied the same way matplotlib will when saving, so
    the reported mm/s and mm/mV match the written PNG.

    Parameters
    ----------
    width_in, height_in
        Figure size in inches.
    dpi
        Dots per inch used when saving.
    duration_s
        Duration of the plotted recording in seconds (full trace width).
    display_mv
        Millivolts spanned by the full vertical axis (default 16 = 4 rows × 4 mV).

    Returns
    -------
    dict
        ``width_px``, ``height_px``, ``width_mm``, ``height_mm``,
        ``speed_mm_s``, ``gain_mm_mv``, ``duration_s``.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if display_mv <= 0:
        raise ValueError("display_mv must be positive.")

    width_px = int(round(width_in * dpi))
    height_px = int(round(height_in * dpi))
    width_mm = (width_px / dpi) * 25.4
    height_mm = (height_px / dpi) * 25.4
    return {
        "width_px": float(width_px),
        "height_px": float(height_px),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "speed_mm_s": width_mm / duration_s,
        "gain_mm_mv": height_mm / display_mv,
        "duration_s": duration_s,
    }


def _resolve_figsize(
    *,
    inches: tuple[float, float] | None,
    resolution: tuple[int, int] | None,
    dpi: int,
) -> tuple[float, float]:
    if inches is not None and resolution is not None:
        raise ValueError("Provide only one of `inches` or `resolution`, not both.")
    if inches is None and resolution is None:
        raise ValueError("Provide `inches=(w, h)` or `resolution=(width_px, height_px)`.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    if resolution is not None:
        width_px, height_px = resolution
        if width_px <= 0 or height_px <= 0:
            raise ValueError("resolution values must be positive.")
        return (width_px / dpi, height_px / dpi)

    width_in, height_in = inches  # type: ignore[misc]
    if width_in <= 0 or height_in <= 0:
        raise ValueError("inches values must be positive.")
    return (float(width_in), float(height_in))


def _format_filter_label(
    *,
    notch_hz: float | None,
    bandpass_hz: tuple[float, float] | None,
    filter_label: str | None,
) -> str | None:
    if filter_label is not None:
        return filter_label
    parts: list[str] = []
    if notch_hz is not None:
        parts.append(str(int(notch_hz) if float(notch_hz).is_integer() else notch_hz))
    if bandpass_hz is not None:
        low, high = bandpass_hz
        low_s = int(low) if float(low).is_integer() else low
        high_s = int(high) if float(high).is_integer() else high
        parts.append(f"{low_s}-{high_s} Hz")
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} ~ {parts[1]}"


def _resolve_footer_id(id: str | None, anonymise: bool) -> str | None:
    if anonymise:
        return ANONYMOUS_ID
    return str(id) if id is not None else None


SUPPORTED_FORMATS = frozenset({"png", "pdf", "svg"})


def _resolve_format(save_path: Path, format: str | None) -> str:
    if format is None:
        suffix = Path(save_path).suffix.lstrip(".").lower()
        if not suffix:
            raise ValueError(
                "Could not infer output format from save_path; "
                "pass format='png'|'pdf'|'svg' or use a matching file extension."
            )
        format = suffix
    fmt = format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {format!r}; expected one of {sorted(SUPPORTED_FORMATS)}."
        )
    return fmt


def plot(
    leads: LeadInput,
    save_path: str | Path,
    *,
    inches: tuple[float, float] | None = None,
    resolution: tuple[int, int] | None = None,
    dpi: int = 300,
    fs: float = 500.0,
    colour: bool = True,
    lead_names: Sequence[str] | None = None,
    layout: Sequence[Sequence[str]] = STANDARD_LAYOUT,
    rhythm_lead: str = RHYTHM_LEAD,
    linewidth: float = 0.8,
    fontname: str = "DejaVu Serif",
    id: str | None = None,
    anonymise: bool = False,
    loc_text: str | None = None,
    filter: bool = False,
    notch_hz: float | None = None,
    bandpass_hz: tuple[float, float] | None = None,
    filter_label: str | None = None,
    extra_footer: str | None = None,
    format: str | None = None,
) -> Path:
    """Plot a 12-lead ECG and save PNG, PDF, or SVG.

    Size may be given as physical ``inches=(w, h)`` or as
    ``resolution=(width_px, height_px)``. Resolution is converted with
    ``inches = pixels / dpi``. Default ``dpi`` is 300.

    Colour mode uses a pink ECG-paper grid; greyscale uses a grey grid.
    PNG colour output is RGB (3 channels); PNG greyscale is mode ``L``
    (1 channel). PDF and SVG are written as vector graphics.

    Paper speed (mm/s) and gain (mm/mV) in the footer are derived from the
    saved figure geometry and the recording duration, so they stay accurate
    when size or DPI change.

    The trace is split into four equal columns. If the sample count is not
    divisible by 4, the last 1–3 samples are dropped and a warning is raised,
    so the columns and the rhythm strip share one time axis. Speed uses that
    drawn duration.

    Parameters
    ----------
    leads
        Lead mapping, ``(12, n)`` / ``(n, 12)`` array, or ``pandas.DataFrame``
        with lead columns (or a single-row frame of lead arrays). Amplitudes
        are expected in millivolts (mV).
    save_path
        Output path. Extension may be ``.png``, ``.pdf``, or ``.svg``.
    inches, resolution
        Mutually exclusive figure size specifications.
    dpi
        Output resolution (controls PNG pixel size; also stored as metadata).
    fs
        Sampling frequency in Hz (used for duration, mm/s, and optional filtering).
    colour
        ``True`` → pink paper; ``False`` → greyscale.
    lead_names
        Names for array inputs (default :data:`LEAD_NAMES`).
    layout
        3×4 lead arrangement.
    rhythm_lead
        Lead used for the continuous bottom strip.
    id, loc_text, extra_footer
        Optional footer strings. ``loc_text`` defaults to a placeholder when
        location metadata is unavailable.
    anonymise
        If ``True``, replace the footer ID with a fixed value
        (``15030416``) so real record IDs cannot leak into ML training images.
    filter
        If ``True``, apply ``notch`` / ``bandpass`` using ``notch_hz`` and
        ``bandpass_hz`` before plotting. If ``False``, those values are only
        used for the footer annotation (signals are left unchanged). Standalone
        filter functions remain available for custom pipelines.
    notch_hz, bandpass_hz, filter_label
        Filter settings and/or footer annotation (e.g. ``60 ~ 0.05-150 Hz``).
        Pass ``filter_label`` to override the auto-built string entirely.
    format
        ``"png"``, ``"pdf"``, or ``"svg"``. If omitted, inferred from
        ``save_path`` suffix.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    # Agg draws into memory and writes a file. pyplot's default backend may
    # try to open a window (Tk), which fails on headless machines and in CI.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    if fs <= 0:
        raise ValueError("fs must be positive.")

    lead_map = as_lead_dict(leads, lead_names=lead_names)
    if filter:
        from ecg_plotter.filters import bandpass, notch

        if notch_hz is not None:
            lead_map = {
                name: notch(sig, fs=fs, freq=notch_hz) for name, sig in lead_map.items()
            }
        if bandpass_hz is not None:
            lowcut, highcut = bandpass_hz
            lead_map = {
                name: bandpass(sig, fs=fs, lowcut=lowcut, highcut=highcut)
                for name, sig in lead_map.items()
            }

    # Four equal columns. Drop at most 3 leftover samples so the panels and
    # the rhythm strip share one time axis, and the footer matches what is drawn.
    n_cols = 4
    n_raw = next(iter(lead_map.values())).shape[0]
    col_len = n_raw // n_cols
    if col_len < 1:
        raise ValueError("Recording is too short to split into a 3×4 layout.")
    n_samples = col_len * n_cols
    dropped = n_raw - n_samples
    if dropped:
        warnings.warn(
            f"Dropped the last {dropped} sample(s) so the four columns stay equal "
            f"({n_raw} samples is not divisible by 4).",
            UserWarning,
            stacklevel=2,
        )
        lead_map = {name: sig[:n_samples] for name, sig in lead_map.items()}

    for row_leads in layout:
        if len(row_leads) != n_cols:
            raise ValueError("Each layout row must contain exactly 4 lead names.")
        for name in row_leads:
            if name not in lead_map:
                raise KeyError(f"Layout lead '{name}' not present in input.")
    if rhythm_lead not in lead_map:
        raise KeyError(f"Rhythm lead '{rhythm_lead}' not present in input.")

    duration_s = n_samples / fs
    width_in, height_in = _resolve_figsize(inches=inches, resolution=resolution, dpi=dpi)
    calib = paper_calibration(width_in, height_in, dpi=dpi, duration_s=duration_s)
    x = arange(n_samples, dtype=float)

    fmt = _resolve_format(Path(save_path), format)
    if fmt == "png":
        from PIL.Image import open as open_image

    if colour:
        face, minor_c, major_c, div_c, trace_c, label_c = (
            _PAPER_FACE,
            _GRID_MINOR,
            _GRID_MAJOR,
            _DIVIDER,
            _TRACE,
            _LABEL,
        )
    else:
        face, minor_c, major_c, div_c, trace_c, label_c = (
            _GREY_FACE,
            _GREY_MINOR,
            _GREY_MAJOR,
            _GREY_DIVIDER,
            "black",
            "black",
        )

    # Row baselines (mV offsets): top lead row → rhythm strip.
    row_offsets = [DISPLAY_MV - ROW_MV * (i + 0.5) for i in range(len(layout))] + [
        ROW_MV * 0.5
    ]

    fig = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor=face)
    FigureCanvasAgg(fig)
    try:
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=face)

        # Lead panels.
        for row_idx, row_leads in enumerate(layout):
            y0 = row_offsets[row_idx]
            for col_idx, name in enumerate(row_leads):
                start = col_idx * col_len
                stop = start + col_len
                segment = lead_map[name][start:stop] + y0
                ax.plot(
                    x[start:stop],
                    segment,
                    color=trace_c,
                    linewidth=linewidth,
                    solid_capstyle="round",
                )
                ax.text(
                    start + col_len * 0.02,
                    y0 + ROW_MV * 0.385,
                    name,
                    fontsize=10,
                    fontname=fontname,
                    color=label_c,
                )

        # Rhythm strip (full width of the samples actually drawn).
        rhythm = lead_map[rhythm_lead] + row_offsets[-1]
        ax.plot(x, rhythm, color=trace_c, linewidth=linewidth, solid_capstyle="round")
        ax.text(
            n_samples * 0.005,
            row_offsets[-1] + ROW_MV * 0.385,
            rhythm_lead,
            fontsize=10,
            fontname=fontname,
            color=label_c,
        )

        # Horizontal dividers between rows (at 12, 8, 4 mV for the default stack).
        for y in (ROW_MV * 3, ROW_MV * 2, ROW_MV):
            ax.plot([0, n_samples], [y, y], color=div_c, linewidth=_DIVIDER_LINEWIDTH)
        # Vertical column walls (lead panels only).
        for boundary in (col_len, 2 * col_len, 3 * col_len):
            ax.plot(
                [boundary, boundary],
                [ROW_MV, DISPLAY_MV],
                color=div_c,
                linewidth=_DIVIDER_LINEWIDTH,
            )

        # Grid: 0.1 mV minor / 0.5 mV major vertically;
        # horizontally, major ≈ 0.2 s (large square) at the given fs.
        samples_per_major = max(fs * 0.2, 1.0)
        samples_per_minor = max(samples_per_major / 5.0, 1.0)
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(samples_per_minor))
        ax.xaxis.set_major_locator(MultipleLocator(samples_per_major))
        ax.grid(which="minor", color=minor_c, linestyle="-", linewidth=0.3, alpha=0.85)
        ax.grid(which="major", color=major_c, linestyle="-", linewidth=0.55, alpha=0.9)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, DISPLAY_MV)
        ax.set_autoscale_on(False)
        ax.tick_params(
            which="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Footer.
        speed = calib["speed_mm_s"]
        gain = calib["gain_mm_mv"]
        filt = _format_filter_label(
            notch_hz=notch_hz, bandpass_hz=bandpass_hz, filter_label=filter_label
        )
        loc = loc_text if loc_text is not None else "LOC  00000-0000"
        calib_text = (
            f"Speed: {speed:.1f} mm/s    "
            f"Limb: {gain:.1f} mm/mV    "
            f"Chest: {gain:.1f} mm/mV"
        )
        if extra_footer:
            calib_text = f"{calib_text}    {extra_footer}"

        footer_id = _resolve_footer_id(id, anonymise)
        footer_y = 0.19
        ax.text(
            n_samples * 0.027, footer_y, loc, fontsize=8, fontname=fontname, color=label_c
        )
        ax.text(
            n_samples * 0.187,
            footer_y,
            calib_text,
            fontsize=8,
            fontname=fontname,
            color=label_c,
        )
        if filt:
            ax.text(
                n_samples * 0.707,
                footer_y,
                filt,
                fontsize=8,
                fontname=fontname,
                color=label_c,
                bbox={"facecolor": "none", "edgecolor": label_c, "boxstyle": "square,pad=0.2"},
            )
        if footer_id is not None:
            ax.text(
                n_samples * 0.939,
                footer_y,
                footer_id,
                fontsize=8,
                fontname=fontname,
                color=label_c,
            )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "png":
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=dpi, facecolor=face, edgecolor="none", pad_inches=0
            )
            buf.seek(0)
            image = open_image(buf)
            if colour:
                image = image.convert("RGB")
            else:
                image = image.convert("L")
            image.save(save_path, format="PNG", dpi=(dpi, dpi))
            buf.close()
        else:
            fig.savefig(
                save_path,
                format=fmt,
                dpi=dpi,
                facecolor=face,
                edgecolor="none",
                pad_inches=0,
            )
    finally:
        # Drop artists so a failed save does not keep a large figure alive.
        fig.clear()

    return save_path.resolve()
