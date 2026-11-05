"""Plot a 12-lead ECG from an explicitly built pandas DataFrame.

The expected wide layout is:

    columns : lead names  (I, II, III, aVR, aVL, aVF, V1..V6)
    rows    : time samples (one sample per row)

Example shape for a 10 s, 500 Hz recording: (5000, 12).
"""

from pathlib import Path

import pandas as pd

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)
    signal = data["leads"]  # dict[str, 1-D array]
    fs = data["fs"]

    # Build the DataFrame explicitly so the column structure is obvious.
    # Each column is one lead; each row is one time sample (mV).
    df = pd.DataFrame(
        {
            "I": signal["I"],
            "II": signal["II"],
            "III": signal["III"],
            "aVR": signal["aVR"],
            "aVL": signal["aVL"],
            "aVF": signal["aVF"],
            "V1": signal["V1"],
            "V2": signal["V2"],
            "V3": signal["V3"],
            "V4": signal["V4"],
            "V5": signal["V5"],
            "V6": signal["V6"],
        }
    )

    print("DataFrame structure:")
    print(f"  shape   : {df.shape}  (n_samples, n_leads)")
    print(f"  columns : {list(df.columns)}")
    print(df.head())

    path = ecg.plot(
        df,
        OUT / "dataframe_colour.png",
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
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
