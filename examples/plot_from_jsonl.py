"""Plot a 12-lead ECG loaded from a synthetic JSON Lines file."""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
JSONL = ROOT / "data/synthetic_data.jsonl"
OUT = ROOT / "output"


def main() -> None:
    # Each line is one recording with lead arrays, required fs, and metadata.
    record = next(ecg.load_jsonl(JSONL))

    path = ecg.plot(
        record["leads"],
        OUT / f"jsonl_{record['id']}.png",
        inches=(9.84, 6.3),
        dpi=300,
        fs=record["fs"],
        colour=True,
        id=str(record["id"]) if record["id"] is not None else None,
        anonymise=True,
        filter=True,
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )
    print(f"Label: {record['label']}")
    print(f"fs: {record['fs']} Hz")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
