"""Non-standard paper sizes -> dynamic footer speed (mm/s) and gain (mm/mV)."""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)
    leads = data["leads"]
    fs = data["fs"]

    common = dict(
        dpi=300,
        fs=fs,
        id=data["record_name"],
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
        colour=True,
    )

    # Classic reference is 9.84 x 6.3 in -> ~25 mm/s and ~10 mm/mV over 10 s.
    # Double width / half height -> ~50 mm/s and ~5 mm/mV.
    # A little narrower and a little taller -> ~20 mm/s and ~12.5 mm/mV.
    # Width stays near 8 in so the footer labels do not collide.
    configs = [
        ("calib_50mms_5mmmV.png", (19.68, 3.15), "50.0 mm/s, 5.0 mm/mV"),
        ("calib_20mms_12p5mmmV.png", (7.872, 7.875), "20.0 mm/s, 12.5 mm/mV"),
    ]

    for filename, inches, label in configs:
        path = ecg.plot(leads, OUT / filename, inches=inches, **common)
        print(f"Saved {path}  ({inches[0]}x{inches[1]} in -> {label})")


if __name__ == "__main__":
    main()
