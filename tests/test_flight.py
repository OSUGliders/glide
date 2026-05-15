from importlib import resources

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import glide.flight as fl
from glide.config import load_config


def get_l2() -> xr.Dataset:
    path = str(resources.files("tests").joinpath("data/flight_test.l2.csv"))
    df = pd.read_csv(path)
    # Restore datetime64 time from integer nanoseconds since epoch
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    return df.set_index("time").to_xarray()


def minimal_conf(overrides: dict | None = None) -> dict:
    """Return a config dict with a simple flight section."""
    conf = load_config()
    conf["flight"] = {
        "rho0": 1025.0,
        "mg": 70.0,
        "Vg": 0.070,
        "Cd0": 0.15,
        "ah": 3.8,
        "calibrate": ["Vg", "mg"],
        "bounds": {"min_pressure": 20.0, "max_pressure": 200.0},
    }
    if overrides:
        conf["flight"].update(overrides)
    return conf


def test_aw_from_geometry():
    aw = fl._aw_from_geometry(aspect_ratio=7.0, sweep_angle=0.7505)
    assert 3.0 < aw < 5.0, f"Unexpected aw={aw}"


def test_cd1_from_params():
    aw = fl._aw_from_geometry(7.0, 0.7505)
    cd1 = fl._cd1_from_params(
        aw, osborne_efficiency=0.8, aspect_ratio=7.0, Cd1_hull=9.7
    )
    assert cd1 > 0


def test_build_params_derives_aw_and_cd1():
    p = fl._build_params({})
    assert p["aw"] is not None
    assert p["Cd1"] is not None


def test_build_params_respects_user_cd1():
    p = fl._build_params({"Cd1": 99.0})
    # User-supplied Cd1 should be preserved (not re-derived).
    assert p["Cd1"] == 99.0


def test_solve_aoa_positive_pitch():
    p = fl._build_params({})
    pitch = np.full(10, np.deg2rad(20.0))
    aoa = fl._solve_aoa(pitch, p["Cd0"], p["Cd1"], p["ah"], p["aw"])
    assert np.all(aoa >= 0)


def test_solve_aoa_negative_pitch():
    p = fl._build_params({})
    pitch = np.full(10, np.deg2rad(-20.0))
    aoa = fl._solve_aoa(pitch, p["Cd0"], p["Cd1"], p["ah"], p["aw"])
    assert np.all(aoa <= 0)


def test_solve_aoa_zero_pitch():
    p = fl._build_params({})
    pitch = np.zeros(5)
    aoa = fl._solve_aoa(pitch, p["Cd0"], p["Cd1"], p["ah"], p["aw"])
    assert np.allclose(aoa, 0.0)


def test_calibrate_returns_all_params():
    ds = get_l2()
    conf = minimal_conf()
    params = fl.calibrate(ds, conf)

    # Every key in DEFAULTS should be present in the output.
    for key in fl.DEFAULTS:
        assert key in params, f"Missing param '{key}' in calibrate output"


def test_calibrate_params_are_finite():
    ds = get_l2()
    conf = minimal_conf()
    params = fl.calibrate(ds, conf)
    for key in ["mg", "Vg", "Cd0", "ah"]:
        assert np.isfinite(params[key]), f"param '{key}' is not finite"


def test_calibrate_bounds_too_restrictive_raises():
    ds = get_l2()
    conf = minimal_conf({"bounds": {"min_pressure": 500.0, "max_pressure": 501.0}})
    with pytest.raises(ValueError, match="Fewer than 100 data points"):
        fl.calibrate(ds, conf)


def test_apply_model_adds_variables():
    ds = get_l2()
    p = fl._build_params({})
    out = fl.apply_model(ds, p)

    for var in (
        "speed_through_water",
        "vertical_glider_velocity",
        "vertical_water_velocity",
        "angle_of_attack",
    ):
        assert var in out, f"'{var}' missing from apply_model output"
        assert out[var].dims == ("time",)


def test_apply_model_stores_global_attrs():
    ds = get_l2()
    p = fl._build_params({})
    out = fl.apply_model(ds, p)

    for key in ["mg", "Vg", "Cd0", "ah", "rho0"]:
        attr = f"flight_model_{key}"
        assert attr in out.attrs, f"Global attribute '{attr}' missing"


def test_apply_model_does_not_mutate_input():
    ds = get_l2()
    original_vars = set(ds.data_vars)
    p = fl._build_params({})
    _ = fl.apply_model(ds, p)
    assert set(ds.data_vars) == original_vars


def test_apply_model_output_shape():
    ds = get_l2()
    p = fl._build_params({})
    out = fl.apply_model(ds, p)
    n = ds.time.size
    assert out["speed_through_water"].shape == (n,)


def test_end_to_end():
    ds = get_l2()
    conf = minimal_conf()
    params = fl.calibrate(ds, conf)
    out = fl.apply_model(ds, params)

    # Calibrated params should be reflected in global attrs.
    assert abs(out.attrs["flight_model_mg"] - params["mg"]) < 1e-9
    assert abs(out.attrs["flight_model_Vg"] - params["Vg"]) < 1e-9

    # Vertical water velocity should be small (order cm/s) for typical data.
    ww = out["vertical_water_velocity"].values
    finite = ww[np.isfinite(ww)]
    assert np.abs(finite).mean() < 0.5, "Mean |ww| suspiciously large"
