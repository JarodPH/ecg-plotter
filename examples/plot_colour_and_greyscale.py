"""Compare inches vs pixel resolution, and colour vs greyscale output."""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)
    leads = data["leads"]
    fs = data["fs"]

    colour_path = ecg.plot(
        leads,
        OUT / "size_colour_inches.png",
        inches=(9.84, 6.3),
        dpi=300,
        fs=fs,
        colour=True,
        id=data["record_name"],
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )

    # Same physical size expressed as pixels: 9.84*300 × 6.3*300
    grey_path = ecg.plot(
        leads,
        OUT / "size_grey_resolution.png",
        resolution=(2952, 1890),
        dpi=300,
        fs=fs,
        colour=False,
        id=data["record_name"],
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )

    print(f"Saved colour (inches)     : {colour_path}")
    print(f"Saved greyscale (pixels)  : {grey_path}")


if __name__ == "__main__":
    main()
