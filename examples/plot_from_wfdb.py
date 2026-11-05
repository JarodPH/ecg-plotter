"""Plot a 12-lead ECG from a synthetic WFDB record."""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)

    path = ecg.plot(
        data["leads"],
        OUT / "wfdb_colour.png",
        inches=(9.84, 6.3),
        dpi=300,
        fs=data["fs"],
        colour=True,
        id=data["record_name"],
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
