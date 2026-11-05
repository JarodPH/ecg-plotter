"""Command-line interface for ecg-plotter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecg_plotter._version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-plotter",
        description="Plot a 12-lead ECG as PNG, PDF, or SVG.",
    )
    parser.add_argument(
        "record",
        nargs="?",
        help="WFDB record path (with or without .hea/.dat), or omit when using --jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output path (.png / .pdf / .svg)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="JSON Lines file of lead arrays (uses --index)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Record index within --jsonl (default: 0)",
    )
    parser.add_argument(
        "--inches",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (mutually exclusive with --resolution)",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH_PX", "HEIGHT_PX"),
        help="Figure size in pixels (mutually exclusive with --inches)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI (default: 300)")
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Sampling frequency in Hz (overrides WFDB/JSONL metadata when set)",
    )
    parser.add_argument(
        "--greyscale",
        action="store_true",
        help="Write greyscale output instead of pink colour paper",
    )
    parser.add_argument(
        "--anonymise",
        action="store_true",
        help="Replace footer ID with the fixed anonymous placeholder",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply notch/bandpass using --notch-hz / --bandpass-hz",
    )
    parser.add_argument(
        "--notch-hz",
        type=float,
        default=None,
        help="Notch frequency in Hz (footer and optional filtering)",
    )
    parser.add_argument(
        "--bandpass-hz",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Band-pass edges in Hz (footer and optional filtering)",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Footer ID text (ignored when --anonymise is set)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default=None,
        help="Output format (default: inferred from --output suffix)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.inches is None and args.resolution is None:
        args.inches = [9.84, 6.3]
    if args.inches is not None and args.resolution is not None:
        parser.error("Provide only one of --inches or --resolution")
    if args.record is None and args.jsonl is None:
        parser.error("Provide a WFDB record path or --jsonl")
    if args.record is not None and args.jsonl is not None:
        parser.error("Provide either a WFDB record path or --jsonl, not both")

    from ecg_plotter.io import load_jsonl, load_wfdb
    from ecg_plotter.sheet import plot

    if args.jsonl is not None:
        if args.index < 0:
            parser.error("--index must be >= 0")
        # Stop once the requested record is found. A research JSONL file can
        # be huge; there is no need to parse every later line.
        chosen = None
        n_seen = 0
        for n_seen, item in enumerate(load_jsonl(args.jsonl), start=1):
            if n_seen - 1 == args.index:
                chosen = item
                break
        if chosen is None:
            if n_seen == 0:
                parser.error(f"No records found in {args.jsonl}")
            parser.error(f"--index {args.index} out of range (0..{n_seen - 1})")
        assert chosen is not None
        leads = chosen["leads"]
        fs = args.fs if args.fs is not None else chosen["fs"]
        record_id = args.id if args.id is not None else (
            str(chosen["id"]) if chosen.get("id") is not None else None
        )
    else:
        data = load_wfdb(args.record)
        leads = data["leads"]
        fs = args.fs if args.fs is not None else data["fs"]
        record_id = args.id if args.id is not None else data["record_name"]

    path = plot(
        leads,
        args.output,
        inches=tuple(args.inches) if args.inches is not None else None,
        resolution=tuple(args.resolution) if args.resolution is not None else None,
        dpi=args.dpi,
        fs=fs,
        colour=not args.greyscale,
        id=record_id,
        anonymise=args.anonymise,
        filter=args.filter,
        notch_hz=args.notch_hz,
        bandpass_hz=tuple(args.bandpass_hz) if args.bandpass_hz is not None else None,
        format=args.format,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
