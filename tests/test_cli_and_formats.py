"""Tests for vector formats, CLI, and packaging metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ecg_plotter import __version__
from ecg_plotter._version import __version__ as version_file
from ecg_plotter.cli import main
from ecg_plotter.sheet import _resolve_format, plot

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"
WFDB_RECORD = EXAMPLES / "synthetic_records" / "synth_0001"
JSONL_PATH = EXAMPLES / "synthetic_data.jsonl"


@pytest.fixture(scope="module")
def wfdb_record_path() -> Path:
    if not WFDB_RECORD.with_suffix(".dat").exists():
        pytest.skip(f"Synthetic WFDB record missing: {WFDB_RECORD}")
    return WFDB_RECORD


@pytest.fixture(scope="module")
def jsonl_path() -> Path:
    if not JSONL_PATH.exists():
        pytest.skip(f"Synthetic JSONL missing: {JSONL_PATH}")
    return JSONL_PATH


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_public_plot_stays_callable_after_other_imports():
    """Importing calibration or the sheet module must not hide ecg.plot."""
    import subprocess
    import sys

    script = (
        "import ecg_plotter.sheet\n"
        "import ecg_plotter as ecg\n"
        "ecg.paper_calibration(1.0, 1.0, dpi=100, duration_s=1.0)\n"
        "assert callable(ecg.plot), repr(ecg.plot)\n"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_version_single_source():
    assert __version__ == version_file
    assert __version__ == "0.1.0"


def test_py_typed_marker_is_shipped():
    marker = Path(__file__).resolve().parents[1] / "ecg_plotter" / "py.typed"
    assert marker.is_file()


# ---------------------------------------------------------------------------
# Format resolution + vector / raster outputs
# ---------------------------------------------------------------------------


def test_resolve_format_from_suffix_and_explicit():
    assert _resolve_format(Path("a.png"), None) == "png"
    assert _resolve_format(Path("a.PDF"), None) == "pdf"
    assert _resolve_format(Path("a.SVG"), None) == "svg"
    assert _resolve_format(Path("a.out"), "svg") == "svg"
    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_format(Path("a.png"), "tiff")
    with pytest.raises(ValueError, match="infer"):
        _resolve_format(Path("noext"), None)


def test_plot_rejects_unsupported_format(leads, fs, tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        plot(
            leads,
            tmp_path / "sheet.tiff",
            resolution=(320, 200),
            dpi=100,
            fs=fs,
        )


def test_plot_pdf_colour_is_valid_pdf(leads, fs, tmp_path):
    path = plot(
        leads,
        tmp_path / "colour.pdf",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=True,
    )
    data = path.read_bytes()
    assert data[:4] == b"%PDF"
    assert path.stat().st_size > 1000


def test_plot_pdf_greyscale_is_valid_pdf(leads, fs, tmp_path):
    path = plot(
        leads,
        tmp_path / "grey.pdf",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=False,
    )
    data = path.read_bytes()
    assert data[:4] == b"%PDF"
    assert path.stat().st_size > 1000


def test_plot_svg_colour_contains_pink_paper(leads, fs, tmp_path):
    path = plot(
        leads,
        tmp_path / "colour.svg",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=True,
    )
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "<svg" in text
    assert "fff5f5" in text
    assert path.stat().st_size > 1000


def test_plot_svg_greyscale_does_not_use_pink_paper(leads, fs, tmp_path):
    path = plot(
        leads,
        tmp_path / "grey.svg",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=False,
    )
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "<svg" in text
    assert "fff5f5" not in text


def test_plot_format_kwarg_overrides_extension(leads, fs, tmp_path):
    # Extension says .bin, but format forces PDF.
    path = plot(
        leads,
        tmp_path / "forced.bin",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        format="pdf",
    )
    assert path.read_bytes()[:4] == b"%PDF"


def test_plot_png_still_respects_channels_after_vector_support(leads, fs, tmp_path):
    colour = plot(
        leads,
        tmp_path / "c.png",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=True,
    )
    grey = plot(
        leads,
        tmp_path / "g.png",
        resolution=(320, 200),
        dpi=100,
        fs=fs,
        colour=False,
    )
    assert Image.open(colour).mode == "RGB"
    assert Image.open(grey).mode == "L"


# ---------------------------------------------------------------------------
# CLI — success paths
# ---------------------------------------------------------------------------


def test_cli_plots_wfdb_to_pdf(wfdb_record_path, tmp_path):
    out = tmp_path / "cli.pdf"
    code = main(
        [
            str(wfdb_record_path),
            "-o",
            str(out),
            "--resolution",
            "320",
            "200",
            "--dpi",
            "100",
            "--anonymise",
            "--filter",
            "--notch-hz",
            "60",
            "--bandpass-hz",
            "0.05",
            "150",
        ]
    )
    assert code == 0
    assert out.is_file()
    assert out.read_bytes()[:4] == b"%PDF"


def test_cli_plots_wfdb_to_png_greyscale(wfdb_record_path, tmp_path):
    out = tmp_path / "cli.png"
    code = main(
        [
            str(wfdb_record_path),
            "-o",
            str(out),
            "--inches",
            "3.2",
            "2.0",
            "--dpi",
            "100",
            "--greyscale",
        ]
    )
    assert code == 0
    image = Image.open(out)
    assert image.mode == "L"
    assert image.size == (320, 200)


def test_cli_plots_wfdb_to_svg(wfdb_record_path, tmp_path):
    out = tmp_path / "cli.svg"
    code = main(
        [
            str(wfdb_record_path),
            "-o",
            str(out),
            "--resolution",
            "320",
            "200",
            "--dpi",
            "100",
            "--format",
            "svg",
        ]
    )
    assert code == 0
    assert "<svg" in out.read_text(encoding="utf-8", errors="ignore").lower()


def test_cli_plots_from_jsonl(jsonl_path, tmp_path):
    out = tmp_path / "from_jsonl.pdf"
    code = main(
        [
            "--jsonl",
            str(jsonl_path),
            "--index",
            "0",
            "-o",
            str(out),
            "--resolution",
            "320",
            "200",
            "--dpi",
            "100",
            "--anonymise",
        ]
    )
    assert code == 0
    assert out.read_bytes()[:4] == b"%PDF"


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI — error paths
# ---------------------------------------------------------------------------


def test_cli_requires_input():
    with pytest.raises(SystemExit):
        main(["-o", "out.png"])


def test_cli_rejects_both_record_and_jsonl(wfdb_record_path, jsonl_path, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                str(wfdb_record_path),
                "--jsonl",
                str(jsonl_path),
                "-o",
                str(tmp_path / "x.pdf"),
            ]
        )


def test_cli_rejects_inches_and_resolution_together(wfdb_record_path, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                str(wfdb_record_path),
                "-o",
                str(tmp_path / "x.pdf"),
                "--inches",
                "3.2",
                "2.0",
                "--resolution",
                "320",
                "200",
            ]
        )


def test_cli_rejects_negative_jsonl_index(jsonl_path, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--jsonl",
                str(jsonl_path),
                "--index",
                "-1",
                "-o",
                str(tmp_path / "x.pdf"),
            ]
        )


def test_cli_rejects_jsonl_index_out_of_range(jsonl_path, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--jsonl",
                str(jsonl_path),
                "--index",
                "9999",
                "-o",
                str(tmp_path / "x.pdf"),
            ]
        )


def test_cli_rejects_invalid_format_choice(wfdb_record_path, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                str(wfdb_record_path),
                "-o",
                str(tmp_path / "x.out"),
                "--format",
                "tiff",
            ]
        )
