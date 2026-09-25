"""Tests for glide.ctd lag and thermal mass corrections."""

from importlib import resources

import gsw
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from scipy.signal import coherence, csd

from glide import config, ctd, process_l1


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
            pressure=("time", np.linspace(0.0, 60.0, n)),
        ),
        coords=dict(time=("time", pd.to_datetime(sci_time, unit="s"))),
    )


def _make_flt(sci, pitch_deg=-26.0):
    """Minimal flight dataset covering the science period."""
    t = np.asarray(sci.time.values).astype("datetime64[ns]").astype("f8") / 1e9
    return xr.Dataset(
        data_vars=dict(
            pitch=("time", np.full(t.size, pitch_deg)),
            lat=("time", np.full(t.size, 44.6)),
        ),
        coords=dict(time=("time", pd.to_datetime(t, unit="s"))),
    )


def _cfg(lag=0.9):
    return {"ctd": {"rbrctd": {"temperature_lag": lag}}}


_TM = ctd.DEFAULTS["rbrctd"]["thermal_mass"]


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


# --- Spectral validation on real RBR legato data ---------------------------
# tests/data/sl1267.ebd.csv holds ~1 h of a real RBR legato deployment
# (2026-05-14 18:00–19:00 UTC, sampled at ~1 Hz). The lag correction aligns
# temperature with conductivity in time, which should drive the C–T
# cross-spectral phase to ~0 at high frequency.

_FS = 1.0  # legato sample rate (Hz)


def _longest_segment(t, *arrays, max_gap=5.0):
    """Return the arrays sliced to the longest gap-free run of ``t`` (gap > max_gap)."""
    brk = np.where(np.diff(t) > max_gap)[0]
    bounds = np.concatenate([[0], brk + 1, [t.size]])
    a, b = max(
        ((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)),
        key=lambda s: s[1] - s[0],
    )
    return (t[a:b], *(x[a:b] for x in arrays))


def _coherent_phase(C, T, fmin=0.05, coh_min=0.5):
    """Mean |cross-spectral phase| of (C, T) over coherent bins above ``fmin``."""
    f, Pxy = csd(C, T, fs=_FS, nperseg=128)
    _, coh = coherence(C, T, fs=_FS, nperseg=128)
    band = (f > fmin) & (coh > coh_min)
    return np.mean(np.abs(np.angle(Pxy[band])))


def _sl1267():
    """Formatted (flight, science) datasets from the sl1267 fixture pair."""
    conf = config.load_config()
    dbd = str(resources.files("tests").joinpath("data/sl1267.dbd.csv"))
    ebd = str(resources.files("tests").joinpath("data/sl1267.ebd.csv"))
    flt = process_l1.apply_qc(
        process_l1.format_l1(process_l1.parse_l1(dbd), conf), conf
    )
    sci = process_l1.apply_qc(
        process_l1.format_l1(process_l1.parse_l1(ebd), conf), conf
    )
    return flt, sci, conf


@pytest.fixture(scope="module")
def rbr_native():
    """(time, T, C) on the native rbrctd grid from the sl1267 fixture."""
    _, sci, _ = _sl1267()
    return ctd._build_native_rbrctd(sci)


def test_fixture_has_rbrctd_variables(rbr_native):
    t, T, C = rbr_native
    assert t.size > 1000  # ~1 h at 1 Hz
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(C))


def test_rbrctd_conductivity_formatted_to_canonical_units():
    # ctd.correct_ctd overwrites the canonical `conductivity` (S/m) with the
    # rbrctd values, so after formatting both must share CF units. The legato
    # reports mS/cm, so a missing conversion shows up as a ~10x scale mismatch.
    ebd = str(resources.files("tests").joinpath("data/sl1267.ebd.csv"))
    conf = config.load_config()
    sci = process_l1.format_l1(process_l1.parse_l1(ebd), conf)

    assert (
        sci["rbrctd_conductivity"].attrs["units"] == sci["conductivity"].attrs["units"]
    )
    ratio = np.nanmedian(sci["rbrctd_conductivity"].values) / np.nanmedian(
        sci["conductivity"].values
    )
    assert 0.8 < ratio < 1.25  # same scale (S/m), not a 10x mS/cm mismatch


def test_lag_removes_high_frequency_ct_phase(rbr_native):
    # Resample the longest gap-free dive segment onto a uniform 1 Hz grid,
    # then compare the C–T phase before and after the configured lag shift.
    t, T, C = _longest_segment(*rbr_native)
    lag = ctd.DEFAULTS["rbrctd"]["temperature_lag"]

    tu = np.arange(t[0], t[-1], 1.0 / _FS)
    Ti = np.interp(tu, t, T)
    Ci = np.interp(tu, t, C)
    Tlag = ctd._apply_lag(tu, Ti, lag)

    before = _coherent_phase(Ci, Ti)
    after = _coherent_phase(Ci, Tlag)

    assert after < 0.3  # ~0 rad in the coherent high-frequency band
    assert after < 0.4 * before  # and a large reduction vs. uncorrected


# --- Thermal mass correction ------------------------------------------------


def test_filter_coefficients_use_speed_in_cm_per_second():
    # The RBR power laws are defined for speed in cm s-1 while glide works in
    # m s-1; dropping the factor of 100 would silently rescale every
    # correction. Pin the coefficients to the cm s-1 convention at U = 0.3 m/s.
    coeffs = _TM["bulk"]
    alpha = coeffs["alpha_prefactor"] * 30.0 ** coeffs["alpha_exponent"]
    tau = coeffs["tau_prefactor"] * 30.0 ** coeffs["tau_exponent"]
    expected_a = 4 * ctd._F_N * alpha * tau / (1 + 4 * ctd._F_N * tau)

    a, b = ctd._filter_coefficients(np.array([0.3]), coeffs)

    np.testing.assert_allclose(a, expected_a, rtol=1e-12)
    np.testing.assert_allclose(b, 1 - 2 * expected_a / alpha, rtol=1e-12)


def test_filter_is_stable_across_the_speed_range():
    U = np.linspace(ctd._U_MIN, ctd._U_MAX, 50)
    for coeffs in _TM.values():
        _, b = ctd._filter_coefficients(U, coeffs)
        assert np.all(np.abs(b) < 1.0)


def test_thermal_mass_zero_for_constant_temperature():
    n = 50
    t = 1.78e9 + np.arange(n, dtype="f8")
    T = np.full(n, 12.0)
    U = np.full(n, 0.3)
    out = ctd._thermal_mass(T, U, t, _TM["bulk"])
    np.testing.assert_allclose(out, 0.0, atol=1e-15)


def test_thermal_mass_responds_to_a_step():
    # A step in temperature should produce a correction of order alpha * step
    # that then decays; the sign follows the direction of the step.
    n = 200
    t = 1.78e9 + np.arange(n, dtype="f8")
    T = np.where(np.arange(n) < 100, 10.0, 11.0)
    U = np.full(n, 0.3)
    coeffs = _TM["short"]
    out = ctd._thermal_mass(T, U, t, coeffs)

    alpha = coeffs["alpha_prefactor"] * 30.0 ** coeffs["alpha_exponent"]
    assert out[100] > 0
    assert 0 < out[100] < alpha  # a < alpha for every stable filter
    assert abs(out[-1]) < abs(out[100])  # and it decays away from the step


def test_thermal_mass_resets_across_gaps():
    n = 40
    t = 1.78e9 + np.arange(n, dtype="f8")
    t[20:] += 10 * ctd._GAP  # a gap far longer than the reset threshold
    T = 10.0 + 0.1 * np.arange(n)
    U = np.full(n, 0.3)
    out = ctd._thermal_mass(T, U, t, _TM["bulk"])
    assert out[20] == 0.0
    assert out[19] != 0.0


def test_correct_thermal_mass_composes_the_three_stages():
    # Bulk correction added, then the long and short stages removed, matching
    # the reference processing this was ported from.
    n = 300
    t = 1.78e9 + np.arange(n, dtype="f8")
    T = 10.0 + np.sin(np.arange(n) / 20.0)
    U = np.full(n, 0.35)
    tm = _TM

    T_bulk = T + ctd._thermal_mass(T, U, t, tm["bulk"])
    expected = (
        T_bulk
        - ctd._thermal_mass(T_bulk, U, t, tm["long"])
        - ctd._thermal_mass(T_bulk, U, t, tm["short"])
    )

    np.testing.assert_allclose(
        ctd._correct_thermal_mass(T, U, t, tm), expected, rtol=1e-12
    )


def test_bound_speed_clips_smooths_and_fills():
    U = np.concatenate(
        [np.full(60, 0.35), [np.nan, np.inf, 0.0, 10.0], np.full(60, 0.35)]
    )
    out = ctd._bound_speed(U)

    assert np.all(np.isfinite(out))
    assert np.all(out >= ctd._U_MIN) and np.all(out <= ctd._U_MAX)
    np.testing.assert_allclose(out[:40], 0.35, atol=1e-12)  # away from the spikes


def test_correct_ctd_adds_temperature_cell_with_flight_data():
    sci = _make_sci()
    out = ctd.correct_ctd(sci, _cfg(), flt=_make_flt(sci))

    assert "temperature_cell" in out
    diff = (out.temperature_cell.values - out.temperature.values)[1:-1]
    assert np.all(np.isfinite(diff))
    assert np.any(diff != 0.0)
    assert np.max(np.abs(diff)) < 1.0  # a small correction, not a rescaling


def test_correct_ctd_without_flight_data_skips_thermal_mass():
    out = ctd.correct_ctd(_make_sci(), _cfg())
    assert "temperature_cell" not in out


def test_correct_ctd_uses_configured_thermal_mass_parameters():
    sci = _make_sci()
    flt = _make_flt(sci)
    zeroed = {stage: dict(alpha_prefactor=0.0) for stage in _TM}
    conf = {"ctd": {"rbrctd": {"thermal_mass": zeroed}}}

    out = ctd.correct_ctd(sci, conf, flt=flt)

    # alpha = 0 means a = 0, so the filter output is zero at every stage.
    np.testing.assert_allclose(
        out.temperature_cell.values, out.temperature.values, atol=1e-12
    )


def test_partial_config_override_keeps_remaining_defaults():
    sci = _make_sci()
    flt = _make_flt(sci)
    default = ctd.correct_ctd(_make_sci(), _cfg(lag=0.9), flt=flt)
    override = ctd.correct_ctd(sci, _cfg(lag=0.5), flt=flt)

    # Only the lag was overridden, so the thermal mass stages still run.
    assert "temperature_cell" in override
    assert not np.allclose(
        override.temperature.values, default.temperature.values, equal_nan=True
    )


def test_shipped_config_overrides_nothing():
    # The shipped config leaves the section empty so the module defaults apply.
    assert config.load_config()["ctd"] == {}


def test_commented_config_block_matches_module_defaults():
    # config.yml documents the defaults in a commented-out block. Uncommenting it
    # must reproduce them exactly, or the file is documenting parameters that are
    # not the ones glide applies.
    text = resources.files("glide").joinpath("assets/config.yml").read_text()
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("# ctd:"))

    block = []
    for ln in lines[start:]:
        if not ln.startswith("#"):
            break
        block.append(ln[2:] if ln.startswith("# ") else ln[1:])

    documented = yaml.safe_load("\n".join(block))
    assert documented["ctd"] == ctd.DEFAULTS


def test_speed_on_real_data_is_plausible():
    flt, sci, _ = _sl1267()
    t, _, _ = ctd._build_native_rbrctd(sci)
    U = ctd._native_speed(sci, flt, t)

    assert np.all(np.isfinite(U))
    assert 0.2 < np.median(U) < 0.5  # a Slocum flies at a few tenths of m s-1


def test_thermal_mass_on_real_data_is_a_small_correction():
    # The legato is a low thermal mass sensor, so the correction should be a
    # few thousandths of a degree: large enough to matter in a sharp
    # thermocline, far too small to be a units or sign error.
    flt, sci, conf = _sl1267()
    out = ctd.correct_ctd(sci, conf, flt=flt)

    diff = out.temperature_cell.values - out.temperature.values
    finite = np.isfinite(diff)
    assert finite.mean() > 0.95  # only the grid edges interpolate to NaN
    diff = diff[finite]
    assert 1e-4 < np.std(diff) < 0.05
    assert np.max(np.abs(diff)) < 0.1


def test_salinity_is_calculated_from_temperature_cell():
    flt, sci, conf = _sl1267()
    sci = ctd.correct_ctd(sci, conf, flt=flt)
    merged = process_l1.calculate_thermodynamics(
        process_l1.merge(flt, sci, conf, "science"), conf
    )

    expected = gsw.SP_from_C(
        10 * merged.conductivity, merged.temperature_cell, merged.pressure
    )
    np.testing.assert_allclose(
        merged.salinity.values, expected.values, atol=1e-6, equal_nan=True
    )


# --- Sea-Bird (pumped ctd41cp) ---------------------------------------------


def _make_sbe_sci(n=200, dt=1.0):
    """Science dataset as a pumped Sea-Bird glider produces it: no rbrctd suite,
    a ctd41cp_time variable, and the CTD data on the canonical variables."""
    t = 1.78e9 + np.arange(n) * dt
    T = 12.0 - 4.0 / (1 + np.exp(-(np.arange(n) - n / 2) / 5.0))  # thermocline
    return xr.Dataset(
        data_vars=dict(
            ctd41cp_time=("time", t - 0.5),
            temperature=("time", T),
            conductivity=("time", np.full(n, 3.4)),
            pressure=("time", 0.2 * np.arange(n)),
        ),
        coords=dict(time=("time", pd.to_datetime(t, unit="s"))),
    )


def _sbe_cfg(alpha=0.03, tau=7.0):
    return {"ctd": {"ctd41cp": {"alpha": alpha, "tau": tau}}}


def test_ctd41cp_selected_by_its_timestamp_variable():
    out = ctd.correct_ctd(_make_sbe_sci(), _sbe_cfg())
    assert "temperature_cell" in out


def test_no_ctd41cp_variable_means_no_correction():
    sci = _make_sbe_sci().drop_vars("ctd41cp_time")
    out = ctd.correct_ctd(sci, _sbe_cfg())
    assert "temperature_cell" not in out


def test_rbrctd_takes_precedence_over_ctd41cp():
    # A dataset carrying both suites is a misconfiguration, but the legato path
    # is the more specific one and must win rather than silently half-applying.
    sci = _make_sci()
    sci["ctd41cp_time"] = ("time", np.asarray(sci.rbrctd_time.values))
    out = ctd.correct_ctd(sci, _cfg(), flt=_make_flt(sci))
    assert "lag-corrected" in out.temperature.attrs["comment"]


def test_cell_thermal_mass_zero_for_constant_temperature():
    t = 1.78e9 + np.arange(50, dtype="f8")
    T = np.full(50, 11.0)
    np.testing.assert_allclose(ctd._cell_thermal_mass(T, t, 0.03, 7.0), T, atol=1e-15)


def test_cell_thermal_mass_warms_the_cell_on_a_cooling_descent():
    # Descending into colder water, the cell walls hold heat, so the water in the
    # cell is warmer than ambient. Sign errors here go straight into salinity.
    t = 1.78e9 + np.arange(100, dtype="f8")
    T = np.where(np.arange(100) < 50, 12.0, 8.0)
    out = ctd._cell_thermal_mass(T, t, 0.03, 7.0)
    assert np.all(out[50:] >= T[50:])
    assert out[50] > T[50]


def test_cell_thermal_mass_uses_nyquist_of_the_sampling_interval():
    # Pin the convention: f = 1/(2*dt), not 1/dt. The two differ by a factor of
    # two in the filter coefficients.
    dt, alpha, tau = 2.0, 0.03, 7.0
    t = 1.78e9 + np.arange(3, dtype="f8") * dt
    T = np.array([10.0, 11.0, 11.0])

    f = 1.0 / (2.0 * dt)
    a = 4 * f * alpha * tau / (1 + 4 * f * tau)
    expected = T[1] - a * (T[1] - T[0])

    out = ctd._cell_thermal_mass(T, t, alpha, tau)
    np.testing.assert_allclose(out[1], expected, rtol=1e-12)


def test_cell_thermal_mass_resets_across_gaps():
    t = 1.78e9 + np.arange(40, dtype="f8")
    t[20:] += 10 * ctd._GAP
    T = 12.0 - 0.1 * np.arange(40)
    out = ctd._cell_thermal_mass(T, t, 0.03, 7.0)
    assert out[20] == T[20]  # correction reset to zero
    assert out[19] != T[19]


def test_correct_ctd41cp_interpolates_onto_the_full_science_grid():
    sci = _make_sbe_sci()
    t41 = sci.ctd41cp_time.values.copy()
    t41[[0, 1]] = 0.0  # sensor silent at the start
    t41[50:60] = 0.0  # and for ten rows in the middle
    sci["ctd41cp_time"] = ("time", t41)

    out = ctd.correct_ctd(sci, _sbe_cfg()).temperature_cell.values

    assert np.all(np.isfinite(out[50:60]))  # interior gap is interpolated over
    assert np.all(np.isnan(out[:2]))  # outside the sensor's span it is NaN


def test_correct_ctd41cp_ignores_sensor_fill_values():
    # Where the sensor has no data it fills temperature with exact zeros and
    # zeroes its own timestamp. Zero degrees passes the QC bounds check, so the
    # timestamp is what keeps a 12 degC step out of the recursion.
    sci = _make_sbe_sci()
    T = sci.temperature.values.copy()
    t41 = sci.ctd41cp_time.values.copy()
    T[100:110] = 0.0
    t41[100:110] = 0.0
    sci["temperature"] = ("time", T)
    sci["ctd41cp_time"] = ("time", t41)

    clean = ctd.correct_ctd(_make_sbe_sci(), _sbe_cfg()).temperature_cell.values
    out = ctd.correct_ctd(sci, _sbe_cfg()).temperature_cell.values

    np.testing.assert_allclose(out[:100], clean[:100], atol=1e-12)
    reported = t41 > ctd._TS_SENTINEL
    assert np.max(np.abs((out - T)[reported])) < 0.1  # no 12 degC step propagated


def test_correct_ctd41cp_does_not_touch_temperature_or_conductivity():
    sci = _make_sbe_sci()
    before = sci.copy(deep=True)
    out = ctd.correct_ctd(sci, _sbe_cfg())
    xr.testing.assert_identical(out.temperature, before.temperature)
    xr.testing.assert_identical(out.conductivity, before.conductivity)


def test_correct_ctd41cp_honours_configured_parameters():
    sci = _make_sbe_sci()
    default = ctd.correct_ctd(_make_sbe_sci(), _sbe_cfg())
    bigger = ctd.correct_ctd(sci, _sbe_cfg(alpha=0.12))

    d_default = np.abs(default.temperature_cell - default.temperature).max()
    d_bigger = np.abs(bigger.temperature_cell - bigger.temperature).max()
    assert d_bigger > 2 * d_default


def test_ctd41cp_on_real_data_is_a_small_correction():
    # sl685 carries a pumped Sea-Bird. The correction should be a few hundredths
    # of a degree through the thermocline: real, but not a rescaling.
    conf = config.load_config()
    ebd = str(resources.files("tests").joinpath("data/sl685.ebd.csv"))
    sci = process_l1.apply_qc(
        process_l1.format_l1(process_l1.parse_l1(ebd), conf), conf
    )
    assert "ctd41cp_time" in sci  # fixture carries the selector
    assert "rbrctd_time" not in sci

    out = ctd.correct_ctd(sci, conf)

    # Only on rows the sensor reported: elsewhere `temperature` is a zero fill,
    # so the difference there reflects the fill, not the correction.
    reported = sci.ctd41cp_time.values > ctd._TS_SENTINEL
    diff = (out.temperature_cell - out.temperature).values[reported]
    diff = diff[np.isfinite(diff)]
    assert diff.size > 0.9 * reported.sum()
    assert 1e-4 < np.std(diff) < 0.05
    assert np.max(np.abs(diff)) < 0.2


def test_cell_thermal_mass_carries_state_through_a_repeated_timestamp():
    # The sensor clock repeats a timestamp occasionally. No time has elapsed, so
    # the correction must carry forward rather than reset to zero.
    t = 1.78e9 + np.array([0.0, 1.0, 1.0, 2.0])
    T = np.array([10.0, 11.0, 11.0, 11.0])

    out = ctd._cell_thermal_mass(T, t, 0.03, 7.0)

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[2] - T[2], out[1] - T[1], rtol=1e-12)
    assert out[1] != T[1]  # and the carried state is not zero


def test_ctd41cp_result_is_not_shifted_by_the_clock_offset():
    # The filter runs on the sensor clock but each value belongs to the row it
    # came from. Interpolating in sensor time would shift every value by the
    # offset between the clocks, which this pins against.
    sci = _make_sbe_sci()
    T = np.asarray(sci.temperature.values, dtype="f8")
    t_sensor = np.asarray(sci.ctd41cp_time.values, dtype="f8")
    expected = ctd._cell_thermal_mass(T, t_sensor, 0.03, 7.0)

    out = ctd.correct_ctd(sci, _sbe_cfg()).temperature_cell.values

    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_ctd41cp_sorts_a_non_monotonic_sensor_clock():
    sci = _make_sbe_sci()
    t41 = np.asarray(sci.ctd41cp_time.values, dtype="f8")
    t41[[80, 81]] = t41[[81, 80]]  # two samples logged out of order
    sci["ctd41cp_time"] = ("time", t41)

    out = ctd.correct_ctd(sci, _sbe_cfg()).temperature_cell.values

    assert np.all(np.isfinite(out))
    # the swap is local: samples well away from it are untouched
    clean = ctd.correct_ctd(_make_sbe_sci(), _sbe_cfg()).temperature_cell.values
    np.testing.assert_allclose(out[:80], clean[:80], atol=1e-12)
