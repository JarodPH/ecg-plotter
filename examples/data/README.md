# Example data

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
