"""Generate synthetic 12-lead ECG example data (no real patient recordings).

Run from the repository root:

    python examples/data/generate_synthetic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import wfdb

from ecg_plotter.leads import LEAD_NAMES

ROOT = Path(__file__).resolve().parent
FS = 500
N_SAMPLES = 5000  # 10 s
DURATION_S = N_SAMPLES / FS

# Distinct synthetic records for demos / JSONL.
RECORDS = (
    {"id": "synth_0001", "label": "Synthetic sinus", "hr_bpm": 72.0, "noise": 0.02},
    {"id": "synth_0002", "label": "Synthetic tachycardia", "hr_bpm": 110.0, "noise": 0.03},
    {"id": "synth_0003", "label": "Synthetic bradycardia", "hr_bpm": 48.0, "noise": 0.02},
)


def _beat_train(t: np.ndarray, hr_bpm: float) -> np.ndarray:
    """Simple synthetic ECG morphology (P + QRS + T), not clinically accurate."""
    rr = 60.0 / hr_bpm
    phase = np.mod(t, rr) / rr
    # Gaussian-like deflections at fixed phases within the RR interval.
    p = 0.12 * np.exp(-0.5 * ((phase - 0.18) / 0.03) ** 2)
    q = -0.08 * np.exp(-0.5 * ((phase - 0.33) / 0.012) ** 2)
    r = 1.00 * np.exp(-0.5 * ((phase - 0.36) / 0.015) ** 2)
    s = -0.18 * np.exp(-0.5 * ((phase - 0.40) / 0.018) ** 2)
    twave = 0.28 * np.exp(-0.5 * ((phase - 0.58) / 0.06) ** 2)
    return p + q + r + s + twave


def make_leads(
    *,
    hr_bpm: float,
    noise: float,
    seed: int,
) -> np.ndarray:
    """Return array shape (n_samples, 12) in :data:`LEAD_NAMES` order, units mV."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES, dtype=float) / FS
    base = _beat_train(t, hr_bpm)

    # Per-lead scales / polarity so the 12-lead sheet is visually varied.
    scales = np.array(
        [1.0, 1.15, 0.85, -0.55, 0.70, 0.45, 0.90, 1.10, 1.25, 1.20, 1.05, 0.95],
        dtype=float,
    )
    phase_shifts = np.linspace(0.0, 0.04, len(LEAD_NAMES))

    cols = []
    for i, (scale, shift) in enumerate(zip(scales, phase_shifts)):
        shifted = _beat_train(t + shift, hr_bpm)
        # Mild lead-specific baseline wander + high-frequency noise.
        wander = 0.05 * np.sin(2 * np.pi * (0.2 + 0.03 * i) * t)
        mains = 0.04 * np.sin(2 * np.pi * 60.0 * t) if i % 3 == 0 else 0.0
        noise_sig = noise * rng.normal(size=N_SAMPLES)
        cols.append(scale * shifted + wander + mains + noise_sig)

    return np.column_stack(cols).astype(np.float64)


def write_wfdb(record_id: str, signal: np.ndarray, label: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        record_name=record_id,
        fs=FS,
        units=["mV"] * 12,
        sig_name=list(LEAD_NAMES),
        p_signal=signal,
        fmt=["16"] * 12,
        adc_gain=[200.0] * 12,
        baseline=[0] * 12,
        comments=[
            "Synthetic ECG for ecg-plotter examples — not real patient data.",
            f"label: {label}",
        ],
        write_dir=str(out_dir),
    )
    return out_dir / record_id


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    records_dir = ROOT / "synthetic_records"
    # Remove previous synthetic tree if regenerating.
    if records_dir.exists():
        for child in records_dir.rglob("*"):
            if child.is_file():
                child.unlink()

    jsonl_rows: list[dict] = []
    csv_lines = ["id,label,path"]

    for i, spec in enumerate(RECORDS):
        signal = make_leads(hr_bpm=spec["hr_bpm"], noise=spec["noise"], seed=1000 + i)
        record_path = write_wfdb(spec["id"], signal, spec["label"], records_dir)

        lead_map = {
            name: signal[:, j].round(6).tolist()
            for j, name in enumerate(LEAD_NAMES)
        }
        jsonl_rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "fs": FS,
                **lead_map,
            }
        )
        csv_lines.append(f"{spec['id']},{spec['label']},{record_path.name}")

    write_jsonl(ROOT / "synthetic_data.jsonl", jsonl_rows)
    (ROOT / "synthetic_records.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    readme = ROOT / "README.md"
    readme.write_text(
        """# Example data

All recordings here are **synthetic** (procedurally generated). They are not
derived from any clinical database and contain no patient information.

| File | Description |
|---|---|
| `synthetic_records/` | WFDB `.hea` / `.dat` pairs (`synth_0001` …) |
| `synthetic_data.jsonl` | Same signals as JSON Lines (one record per line; includes ``fs``) |
| `synthetic_records.csv` | Index of id / label / record name |
| `generate_synthetic.py` | Regenerates the files above |

Sampling rate: 500 Hz, duration: 10 s (5000 samples), 12 leads in standard
order (aVL before aVF). ``load_wfdb`` accepts any channel order from the WFDB
header.
""",
        encoding="utf-8",
    )

    print(f"Wrote {len(RECORDS)} synthetic WFDB records to {records_dir}")
    print(f"Wrote {ROOT / 'synthetic_data.jsonl'}")


if __name__ == "__main__":
    main()
