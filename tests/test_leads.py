"""Tests for lead normalisation and input validation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ecg_plotter.leads import LEAD_NAMES, as_lead_dict


def test_as_lead_dict_from_mapping(leads):
    out = as_lead_dict(leads)
    assert set(out) == set(LEAD_NAMES)
    assert all(out[name].shape == (len(leads["I"]),) for name in LEAD_NAMES)
    assert out["II"].dtype == float


def test_as_lead_dict_from_array_leads_first(leads_array):
    out = as_lead_dict(leads_array)
    assert list(out) == list(LEAD_NAMES)
    np.testing.assert_allclose(out["I"], leads_array[0])


def test_as_lead_dict_from_array_samples_first(leads_array):
    out = as_lead_dict(leads_array.T)
    assert list(out) == list(LEAD_NAMES)
    np.testing.assert_allclose(out["V6"], leads_array[-1])


def test_as_lead_dict_from_dataframe(leads_df):
    out = as_lead_dict(leads_df)
    assert list(out) == list(LEAD_NAMES)
    np.testing.assert_allclose(out["aVL"], leads_df["aVL"].to_numpy())


def test_as_lead_dict_from_single_row_array_cells(leads):
    row = pd.DataFrame([{name: leads[name].tolist() for name in LEAD_NAMES}])
    out = as_lead_dict(row)
    assert len(out["I"]) == len(leads["I"])
    np.testing.assert_allclose(out["III"], leads["III"])


def test_empty_mapping_raises():
    with pytest.raises(ValueError, match="empty"):
        as_lead_dict({})


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        as_lead_dict(pd.DataFrame())


def test_unequal_lead_lengths_raises(leads):
    bad = dict(leads)
    bad["V1"] = leads["V1"][:-10]
    with pytest.raises(ValueError, match="same length"):
        as_lead_dict(bad)


def test_missing_dataframe_columns_raises(leads_df):
    incomplete = leads_df.drop(columns=["V3", "aVR"])
    with pytest.raises(KeyError, match="missing lead columns"):
        as_lead_dict(incomplete, lead_names=LEAD_NAMES)


def test_dataframe_unknown_columns_raises():
    df = pd.DataFrame({"foo": np.zeros(10), "bar": np.zeros(10)})
    with pytest.raises(ValueError, match="Could not infer lead columns"):
        as_lead_dict(df)


def test_unlabelled_twelve_column_dataframe_warns(leads_array):
    # Columns 0..11, not I, II, ... The first column must become lead I.
    df = pd.DataFrame(leads_array.T)
    with pytest.warns(UserWarning, match="not lead names"):
        out = as_lead_dict(df)
    assert list(out) == list(LEAD_NAMES)
    np.testing.assert_allclose(out["I"], leads_array[0])
    np.testing.assert_allclose(out["V6"], leads_array[-1])


def test_named_dataframe_does_not_warn_about_lead_order(leads_df):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = as_lead_dict(leads_df)
    assert list(out) == list(LEAD_NAMES)


def test_array_wrong_shape_raises():
    with pytest.raises(ValueError, match="2-D"):
        as_lead_dict(np.zeros(12))
    with pytest.raises(ValueError, match="dimension of size 12"):
        as_lead_dict(np.zeros((5, 100)))
