"""Plot a 12-lead ECG as vector PDF and SVG."""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)
    common = dict(
        inches=(9.84, 6.3),
        dpi=300,
        fs=data["fs"],
        id=data["record_name"],
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )

    # Format is inferred from the file extension (or pass format="pdf" / "svg").
    pdf_path = ecg.plot(
        data["leads"],
        OUT / "synth_colour.pdf",
        colour=True,
        **common,
    )
    svg_path = ecg.plot(
        data["leads"],
        OUT / "synth_greyscale.svg",
        colour=False,
        **common,
    )

    print(f"Saved PDF : {pdf_path}")
    print(f"Saved SVG : {svg_path}")


if __name__ == "__main__":
    main()
