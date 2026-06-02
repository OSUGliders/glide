"""Tests for glide.ctd lag correction."""

import numpy as np
import pandas as pd
import xarray as xr

from glide import ctd


def _make_sci(t_native=None, T=None, C=None, sci_time=None):
    """Build a minimal science dataset with rbrctd_* plus canonical T and C."""
    n = 100
    if sci_time is None:
        sci_time = 1.78e9 + np.arange(n, dtype="f8")  # well above sentinel
    if t_native is None:
        t_native = sci_time + 0.1
    if T is None:
        T = 10.0 + np.linspace(0, 5, n).astype("f8")
    if C is None:
        C = 35.0 + np.linspace(0, 0.5, n).astype("f8")

    return xr.Dataset(
        data_vars=dict(
            rbrctd_time=("time", t_native),
            rbrctd_temperature=("time", T),
            rbrctd_conductivity=("time", C),
            temperature=("time", T.copy()),
            conductivity=("time", C.copy()),
        ),
        coords=dict(time=("time", pd.to_datetime(sci_time, unit="s"))),
    )


def _cfg(lag=0.9):
    return {"ctd": {"rbrctd": {"temperature_lag": lag}}}


def test_correct_ctd_no_rbrctd_returns_unchanged():
    sci = _make_sci().drop_vars(
        ["rbrctd_time", "rbrctd_temperature", "rbrctd_conductivity"]
    )
    sci_in = sci.copy(deep=True)
    out = ctd.correct_ctd(sci, _cfg())
    xr.testing.assert_identical(out.temperature, sci_in.temperature)
    xr.testing.assert_identical(out.conductivity, sci_in.conductivity)


def test_apply_lag_zero_is_identity():
    t = np.arange(10, dtype="f8")
    T = np.sin(t).astype("f8")
    np.testing.assert_array_equal(ctd._apply_lag(t, T, 0.0), T)


def test_apply_lag_on_linear_ramp_shifts_in_time():
    # Linear ramp T = t + 100. Shifting time by -lag and interpolating back
    # gives T(t - lag) = t - lag + 100 well inside the grid; edges flatten
    # because of np.interp's left/right clamping.
    t = np.arange(20, dtype="f8")
    T = t + 100.0
    lag = 0.5
    out = ctd._apply_lag(t, T, lag)
    np.testing.assert_allclose(out[5:-5], T[5:-5] + lag, atol=1e-10)


def test_build_native_rbrctd_drops_sentinel_and_sorts():
    sci_time = 1.78e9 + np.array([2.0, 0.0, 1.0])
    t_native = np.array([1.0, 1.78e9 + 5.0, 1.78e9 + 6.0])  # first is sentinel
    sci = xr.Dataset(
        data_vars=dict(
            rbrctd_time=("time", t_native),
            rbrctd_temperature=("time", np.array([10.0, 11.0, 12.0])),
            rbrctd_conductivity=("time", np.array([30.0, 31.0, 32.0])),
        ),
        coords=dict(time=("time", pd.to_datetime(sci_time, unit="s"))),
    )
    t, T, C = ctd._build_native_rbrctd(sci)
    assert t.size == 2
    np.testing.assert_array_equal(t, [1.78e9 + 5.0, 1.78e9 + 6.0])
    np.testing.assert_array_equal(T, [11.0, 12.0])
    np.testing.assert_array_equal(C, [31.0, 32.0])


def test_build_native_rbrctd_dedup_on_time():
    sci_time = 1.78e9 + np.arange(3, dtype="f8")
    t_native = 1.78e9 + np.array([10.0, 10.0, 11.0])
    sci = xr.Dataset(
        data_vars=dict(
            rbrctd_time=("time", t_native),
            rbrctd_temperature=("time", np.array([10.0, 99.0, 11.0])),
            rbrctd_conductivity=("time", np.array([30.0, 31.0, 32.0])),
        ),
        coords=dict(time=("time", pd.to_datetime(sci_time, unit="s"))),
    )
    t, _, _ = ctd._build_native_rbrctd(sci)
    assert t.size == 2  # duplicate dropped


def test_correct_ctd_replaces_temperature_with_lag_shift():
    # On the native grid T = T0 + slope*(t - t0). After a lag shift by `lag`
    # seconds, the value at time t should equal T sampled at (t + lag).
    n = 100
    t_native = 1.78e9 + np.arange(n, dtype="f8")
    sci_time = t_native.copy()
    slope = 0.1
    T = 10.0 + slope * np.arange(n)
    sci = _make_sci(t_native=t_native, T=T, sci_time=sci_time)

    out = ctd.correct_ctd(sci, _cfg(lag=0.9))

    interior = slice(10, -10)
    np.testing.assert_allclose(
        out.temperature.values[interior], T[interior] + slope * 0.9, atol=1e-9
    )


def test_correct_ctd_replaces_conductivity_unadjusted():
    # Conductivity is reinterpolated from the native grid but NOT lag-shifted.
    # With sci_time == t_native, the values should round-trip unchanged.
    n = 100
    t_native = 1.78e9 + np.arange(n, dtype="f8")
    sci_time = t_native.copy()
    C = 35.0 + 0.01 * np.arange(n)
    sci = _make_sci(t_native=t_native, C=C, sci_time=sci_time)

    out = ctd.correct_ctd(sci, _cfg(lag=0.9))

    np.testing.assert_allclose(out.conductivity.values, C, atol=1e-12)


def test_correct_ctd_preserves_temperature_attrs():
    sci = _make_sci()
    sci["temperature"].attrs = {
        "long_name": "Temperature",
        "units": "celsius",
        "valid_min": -5.0,
        "valid_max": 50.0,
    }
    out = ctd.correct_ctd(sci, _cfg())
    assert out.temperature.attrs["long_name"] == "Temperature"
    assert out.temperature.attrs["units"] == "celsius"
    assert out.temperature.attrs["valid_min"] == -5.0
    assert "ctd.correct_ctd" in out.temperature.attrs["comment"]


def test_correct_ctd_too_few_samples_returns_unchanged():
    sci = _make_sci()
    bad = np.full_like(sci["rbrctd_time"].values, np.nan)
    bad[0] = 1.78e9 + 5.0
    sci["rbrctd_time"] = ("time", bad)
    sci_in = sci.copy(deep=True)
    out = ctd.correct_ctd(sci, _cfg())
    xr.testing.assert_identical(out.temperature, sci_in.temperature)
    xr.testing.assert_identical(out.conductivity, sci_in.conductivity)
