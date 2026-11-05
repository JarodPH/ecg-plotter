"""Filter leads yourself, then plot with ``filter=False``.

``notch_hz`` / ``bandpass_hz`` still appear in the footer; they are not
re-applied because filtering was done up front. Use this pattern for
median baseline removal or any custom chain.
"""

from pathlib import Path

import ecg_plotter as ecg

ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "data/synthetic_records/synth_0001"
OUT = ROOT / "output"


def main() -> None:
    data = ecg.load_wfdb(RECORD)
    leads = data["leads"]
    fs = data["fs"]

    # Standalone filters — chain only what you want.
    leads = ecg.apply_to_leads(leads, ecg.notch, fs=fs, freq=60.0)
    leads = ecg.apply_to_leads(leads, ecg.bandpass, fs=fs, lowcut=0.05, highcut=150.0)
    # Optional extra:
    # leads = ecg.apply_to_leads(leads, ecg.median_baseline, kernel_size=201)

    path = ecg.plot(
        leads,
        OUT / "standalone_filters.png",
        inches=(9.84, 6.3),
        dpi=300,
        fs=fs,
        colour=True,
        id=data["record_name"],
        anonymise=True,
        filter=False,  # already filtered; kwargs are footer text only
        notch_hz=60,
        bandpass_hz=(0.05, 150),
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
