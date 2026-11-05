"""Plot at 150 DPI — half the default 300, smaller pixel dimensions."""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)
    # Same physical paper as the classic sheet (9.84 x 6.3 in), but 150 DPI
    # -> 1476 x 945 px instead of 2952 x 1890 at 300 DPI. Speed/gain stay ~25/10.
    path = ecg.plot(
        data["leads"],
        OUT / "size_low_dpi_150.png",
        inches=(9.84, 6.3),
        dpi=150,
        fs=data["fs"],
        colour=True,
        id=data["record_name"],
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )
    print(f"Saved {path} (9.84x6.3 in @ 150 DPI -> 1476x945 px)")


if __name__ == "__main__":
    main()
