"""Tests for WFDB / JSONL loaders and lead stacking — including synthetic examples."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from numpy import allclose, asarray

from ecg_plotter.io import leads_to_array, load_jsonl, load_wfdb
from ecg_plotter.leads import LEAD_NAMES

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"
WFDB_RECORD = EXAMPLES / "synthetic_records" / "synth_0001"
JSONL_PATH = EXAMPLES / "synthetic_data.jsonl"


@pytest.fixture(scope="module")
def wfdb_data():
    if not WFDB_RECORD.with_suffix(".dat").exists():
        pytest.skip(f"Synthetic WFDB record missing: {WFDB_RECORD}")
    return load_wfdb(WFDB_RECORD)


@pytest.fixture(scope="module")
def jsonl_records():
    if not JSONL_PATH.exists():
        pytest.skip(f"Synthetic JSONL missing: {JSONL_PATH}")
    return list(load_jsonl(JSONL_PATH))


# ---------------------------------------------------------------------------
# leads_to_array
# ---------------------------------------------------------------------------


def test_leads_to_array_shape_and_order(leads):
    arr = leads_to_array(leads)
    assert arr.shape == (12, len(leads["I"]))
    assert allclose(arr[0], leads["I"])
    assert allclose(arr[5], leads["aVF"])  # LEAD_NAMES index of aVF
    assert list(LEAD_NAMES).index("aVF") == 5


def test_leads_to_array_custom_order(leads):
    arr = leads_to_array(leads, lead_names=("V1", "II"))
    assert arr.shape == (2, len(leads["I"]))
    assert allclose(arr[0], leads["V1"])
    assert allclose(arr[1], leads["II"])


def test_leads_to_array_missing_lead_raises(leads):
    incomplete = {k: v for k, v in leads.items() if k != "V4"}
    with pytest.raises(KeyError, match="Missing leads"):
        leads_to_array(incomplete)


# ---------------------------------------------------------------------------
# load_wfdb — synthetic example record
# ---------------------------------------------------------------------------


def test_load_wfdb_metadata_and_leads(wfdb_data):
    assert wfdb_data["record_name"] == "synth_0001"
    assert wfdb_data["fs"] == 500.0
    assert wfdb_data["n_samples"] == 5000
    assert wfdb_data["sig_name"] == list(LEAD_NAMES)
    assert set(wfdb_data["leads"]) == set(LEAD_NAMES)
    for name, sig in wfdb_data["leads"].items():
        assert sig.shape == (5000,)
        assert sig.dtype.kind == "f"
    # Header order is authoritative (synthetic uses aVL before aVF).
    assert wfdb_data["sig_name"].index("aVL") < wfdb_data["sig_name"].index("aVF")
    assert any("Synthetic" in c for c in wfdb_data["record"].comments)


def test_load_wfdb_accepts_hea_suffix():
    data = load_wfdb(WFDB_RECORD.with_suffix(".hea"))
    assert data["record_name"] == "synth_0001"
    assert data["n_samples"] == 5000


def test_load_wfdb_accepts_dat_suffix():
    data = load_wfdb(WFDB_RECORD.with_suffix(".dat"))
    assert data["fs"] == 500.0


def test_load_wfdb_lead_names_override_wrong_count_raises():
    with pytest.raises(ValueError, match="does not match"):
        load_wfdb(WFDB_RECORD, lead_names=("I", "II"))


def test_load_wfdb_roundtrip_with_leads_to_array(wfdb_data):
    arr = leads_to_array(wfdb_data["leads"], lead_names=LEAD_NAMES)
    assert arr.shape == (12, 5000)
    assert allclose(arr[0], wfdb_data["leads"]["I"])


def test_load_wfdb_missing_path_raises():
    with pytest.raises(Exception):
        load_wfdb(EXAMPLES / "synthetic_records" / "does_not_exist")


def test_load_wfdb_respects_header_order_avf_before_avl(tmp_path):
    """Any WFDB channel order is fine — names come from the header."""
    import wfdb
    from numpy import zeros

    # Alternate header order used by some PhysioNet / hospital exports.
    alt_order = (
        "I",
        "II",
        "III",
        "aVR",
        "aVF",
        "aVL",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    record_name = "alt_order"
    wfdb.wrsamp(
        record_name=record_name,
        fs=500,
        units=["mV"] * 12,
        sig_name=list(alt_order),
        p_signal=zeros((100, 12)),
        fmt=["16"] * 12,
        adc_gain=[200.0] * 12,
        baseline=[0] * 12,
        write_dir=str(tmp_path),
    )
    data = load_wfdb(tmp_path / record_name)
    assert data["sig_name"] == list(alt_order)
    assert data["sig_name"].index("aVF") < data["sig_name"].index("aVL")
    assert set(data["leads"]) == set(LEAD_NAMES)


def test_load_wfdb_warns_when_units_not_mv(tmp_path):
    import wfdb
    from numpy import zeros

    record_name = "units_uV"
    signal = zeros((100, 12))
    wfdb.wrsamp(
        record_name=record_name,
        fs=500,
        units=["uV"] * 12,
        sig_name=list(LEAD_NAMES),
        p_signal=signal,
        fmt=["16"] * 12,
        adc_gain=[1.0] * 12,
        baseline=[0] * 12,
        write_dir=str(tmp_path),
    )
    with pytest.warns(UserWarning, match="non-mV units"):
        data = load_wfdb(tmp_path / record_name)
    assert data["units"] == ["uV"] * 12


def test_warn_if_units_not_mv_silent_for_mv():
    from ecg_plotter.io import _warn_if_units_not_mv

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_units_not_mv(["mV"] * 12, Path("dummy"))
        _warn_if_units_not_mv(["mV", "millivolt"], Path("dummy"))
    assert caught == []


def test_load_wfdb_synthetic_units_are_mv(wfdb_data):
    assert wfdb_data["units"] == ["mV"] * 12


# ---------------------------------------------------------------------------
# load_jsonl — synthetic example file + error paths
# ---------------------------------------------------------------------------


def test_load_jsonl_reads_all_records(jsonl_records):
    assert len(jsonl_records) == 3
    first = jsonl_records[0]
    assert first["id"] == "synth_0001"
    assert first["label"] == "Synthetic sinus"
    assert first["fs"] == 500.0
    assert set(LEAD_NAMES).issubset(first["leads"])
    n = len(first["leads"]["I"])
    assert n == 5000
    for name in LEAD_NAMES:
        assert first["leads"][name].shape == (n,)


def test_load_jsonl_all_records_have_consistent_leads(jsonl_records):
    for record in jsonl_records:
        lengths = {name: len(sig) for name, sig in record["leads"].items()}
        assert len(set(lengths.values())) == 1
        assert record["id"] is not None
        assert record["label"] is not None
        assert record["fs"] > 0
        assert "meta" in record
        assert "fs" in record["meta"]


def test_load_jsonl_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        list(load_jsonl(path))


def test_load_jsonl_non_object_raises(tmp_path):
    path = tmp_path / "list.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected object"):
        list(load_jsonl(path))


def test_load_jsonl_no_leads_raises(tmp_path):
    path = tmp_path / "noleads.jsonl"
    path.write_text(json.dumps({"id": 1, "label": "x", "fs": 500}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No lead arrays"):
        list(load_jsonl(path))


def test_load_jsonl_missing_fs_raises(tmp_path):
    lead = {name: [0.0, 0.1, 0.0, -0.1] for name in LEAD_NAMES}
    path = tmp_path / "nofs.jsonl"
    path.write_text(json.dumps({"id": 1, "label": "x", **lead}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required field 'fs'"):
        list(load_jsonl(path))


def test_load_jsonl_null_fs_raises(tmp_path):
    lead = {name: [0.0, 0.1, 0.0, -0.1] for name in LEAD_NAMES}
    path = tmp_path / "nullfs.jsonl"
    path.write_text(
        json.dumps({"id": 1, "label": "x", "fs": None, **lead}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing required field 'fs'"):
        list(load_jsonl(path))


def test_load_jsonl_nonpositive_fs_raises(tmp_path):
    lead = {name: [0.0, 0.1, 0.0, -0.1] for name in LEAD_NAMES}
    path = tmp_path / "badfs.jsonl"
    path.write_text(
        json.dumps({"id": 1, "label": "x", "fs": 0, **lead}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'fs' must be positive"):
        list(load_jsonl(path))


def test_load_jsonl_skips_blank_lines(tmp_path):
    lead = {name: [0.0, 0.1, 0.0, -0.1] for name in LEAD_NAMES}
    path = tmp_path / "blank.jsonl"
    path.write_text(
        "\n" + json.dumps({"id": 7, "label": "demo", "fs": 250, **lead}) + "\n\n",
        encoding="utf-8",
    )
    records = list(load_jsonl(path))
    assert len(records) == 1
    assert records[0]["id"] == 7
    assert records[0]["fs"] == 250.0
    assert allclose(records[0]["leads"]["I"], asarray([0.0, 0.1, 0.0, -0.1]))
