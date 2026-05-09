# Steady-state glider flight model following Merckelbach et al. (2010)

import logging

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.optimize import fmin as fminimize
from scipy.optimize import fsolve

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model parameters
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    rho0=1025.0,  # reference density (kg m-3)
    mg=70.0,  # glider mass (kg)
    Vg=0.070,  # glider volume (m3)
    Cd0=0.15,  # parasitic drag coefficient (-)
    ah=3.8,  # hull lift coefficient slope (rad-1)
    Cd1=10.0,  # induced drag coefficient (s-2); derived from aw if not set
    aw=None,  # wing lift coefficient slope (rad-1); derived from AR/Omega
    S=0.10,  # reference area (m-2)
    AR=7.0,  # aspect ratio
    Omega=0.7505,  # sweep angle (rad ≈ 43 degrees)
    eOsborne=0.8,  # Osborne efficiency factor
    Cd1_hull=9.7,  # hull induced-drag coefficient
    epsilon=5e-10,  # hull compressibility (Pa-1)
)

# Scale factors used internally by the optimiser to keep parameters near O(1).
_SCALE = dict(Cd0=0.15, Vg=0.070, mg=70.0, ah=3.8, Cd1=10.0, aw=4.0)

# Physical constants
_G = 9.81  # m s-2


def _aw_from_geometry(AR: float, Omega: float) -> float:
    """Lift-curve slope from wing aspect ratio and sweep angle."""
    return 2 * np.pi * AR / (2 + np.sqrt(AR**2 * (1 + np.tan(Omega) ** 2) + 4))


def _cd1_from_params(aw: float, eOsborne: float, AR: float, Cd1_hull: float) -> float:
    """Induced drag coefficient from wing and hull contributions."""
    Cd1_wing = aw**2 / (np.pi * eOsborne * AR)
    return Cd1_wing + Cd1_hull


def _build_params(cfg: dict) -> dict:
    """Merge user config over defaults, derive aw / Cd1 if not explicitly set."""
    p = {**DEFAULTS, **cfg}

    if p["aw"] is None:
        p["aw"] = _aw_from_geometry(p["AR"], p["Omega"])
        _log.debug("Derived aw = %.4f from AR and Omega", p["aw"])

    # Cd1 is derived unless the caller overrides it directly.
    if "Cd1" not in cfg:
        p["Cd1"] = _cd1_from_params(p["aw"], p["eOsborne"], p["AR"], p["Cd1_hull"])
        _log.debug("Derived Cd1 = %.4f", p["Cd1"])

    return p


def _compute_buoyancy(
    pressure_Pa: np.ndarray, rho: np.ndarray, Vbp_m3: np.ndarray, p: dict
) -> tuple[np.ndarray, float]:
    """Return buoyancy force FB (N) and gravity force Fg (N)."""
    FB = _G * rho * (p["Vg"] * (1.0 - p["epsilon"] * pressure_Pa) + Vbp_m3)
    Fg = p["mg"] * _G
    return FB, Fg


def _aoa_equation(alpha: float, pitch_rad: float, p: dict) -> float:
    """Implicit equation whose zero is the angle of attack."""
    ah_aw = p["ah"] + p["aw"]
    return (p["Cd0"] + p["Cd1"] * alpha**2) / (
        ah_aw * np.tan(alpha + pitch_rad)
    ) - alpha


def _solve_aoa(pitch_rad: np.ndarray, p: dict) -> np.ndarray:
    """Solve for angle of attack at each pitch sample.

    Builds an interpolating table over the pitch range for efficiency, then
    applies it.  Pitches whose magnitude is below ~5° receive aoa = 0.
    """
    pitch_abs = np.abs(pitch_rad)
    pitch_grid = np.linspace(5 * np.pi / 180, 60 * np.pi / 180, 120)

    aoa_grid = np.zeros_like(pitch_grid)
    for i, _pitch in enumerate(pitch_grid):
        sol = fsolve(_aoa_equation, _pitch / 50.0, args=(_pitch, p), full_output=True)
        aoa_grid[i] = sol[0][0]

    ifun = interp1d(pitch_grid, aoa_grid, bounds_error=False, fill_value=0.0)

    aoa = np.where(
        pitch_abs >= pitch_grid[0],
        ifun(pitch_abs) * np.sign(pitch_rad),
        0.0,
    )
    return aoa


def _compute_speed(
    FB: np.ndarray,
    Fg: float,
    rho: np.ndarray,
    alpha: np.ndarray,
    pitch_rad: np.ndarray,
    p: dict,
) -> np.ndarray:
    """Incident water speed U (m s-1) from force balance."""
    numer = 2.0 * (FB - Fg) * np.sin(pitch_rad + alpha)
    denom = rho * p["S"] * (p["Cd0"] + p["Cd1"] * alpha**2)
    U2 = np.where(denom != 0, numer / denom, 0.0)
    U2 = np.maximum(U2, 0.0)
    return np.sqrt(U2)


def _solve_flight_si(
    pressure_Pa: np.ndarray,
    dzdt: np.ndarray,
    pitch_rad: np.ndarray,
    Vbp_m3: np.ndarray,
    density: np.ndarray,
    p: dict,
) -> dict:
    """Core steady-state flight model operating entirely in SI units.

    Parameters
    ----------
    pressure_Pa : array
        Pressure in Pa (used only for buoyancy / compressibility).
    dzdt : array
        Observed vertical velocity dz/dt in m s⁻¹ (upward positive),
        pre-computed from np.gradient(z_m, time_s) on the full timeseries.
    pitch_rad : array
        Pitch in radians (positive nose-up).
    Vbp_m3 : array
        Ballast volume change in m³.
    density : array
        In-situ seawater density in kg m⁻³.
    p : dict
        Model parameters (see DEFAULTS).

    Returns
    -------
    dict with keys: speed_through_water (m s⁻¹), vertical_glider_velocity
    (m s⁻¹), vertical_water_velocity (m s⁻¹), angle_of_attack (degrees).
    """
    FB, Fg = _compute_buoyancy(pressure_Pa, density, Vbp_m3, p)
    alpha = _solve_aoa(pitch_rad, p)
    U = _compute_speed(FB, Fg, density, alpha, pitch_rad, p)

    # wg: vertical component of glider velocity through the water (upward +)
    # sin(pitch + alpha) is negative when diving (negative pitch), positive
    # when climbing — matching the z convention.
    wg = U * np.sin(pitch_rad + alpha)
    ww = dzdt - wg  # vertical water velocity

    return dict(
        speed_through_water=U,
        vertical_glider_velocity=wg,
        vertical_water_velocity=ww,
        angle_of_attack=np.rad2deg(alpha),
    )


def solve_flight(
    time_s: np.ndarray,
    pressure_dbar: np.ndarray,
    z_m: np.ndarray,
    pitch_deg: np.ndarray,
    ballast_cm3: np.ndarray,
    density: np.ndarray,
    p: dict,
) -> dict:
    """Run the steady-state flight model.

    Accepts glide-native units and converts to SI before running the model.

    Parameters
    ----------
    time_s : array
        Time in seconds since epoch.
    pressure_dbar : array
        Pressure in dbar (used only for buoyancy / compressibility).
    z_m : array
        Height in metres (upward positive, from gsw.z_from_p).
    pitch_deg : array
        Pitch in degrees (positive nose-up).
    ballast_cm3 : array
        Ballast pumped in cm³.
    density : array
        In-situ seawater density in kg m⁻³.
    p : dict
        Model parameters (see DEFAULTS).

    Returns
    -------
    dict with keys: speed_through_water (m s⁻¹), vertical_glider_velocity
    (m s⁻¹), vertical_water_velocity (m s⁻¹), angle_of_attack (degrees).
    """
    # --- unit conversion: glide native → SI -----------------------------------
    pressure_Pa = pressure_dbar * 1e4  # dbar → Pa
    pitch_rad = np.deg2rad(pitch_deg)  # degrees → radians
    Vbp_m3 = ballast_cm3 * 1e-6  # cm³ → m³
    # z_m, density already in SI; time already in seconds
    # --------------------------------------------------------------------------

    dzdt = np.gradient(z_m, time_s)  # dz/dt on full timeseries (upward +)
    return _solve_flight_si(pressure_Pa, dzdt, pitch_rad, Vbp_m3, density, p)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _cost(
    x: np.ndarray,
    param_names: list[str],
    base_params: dict,
    pressure_Pa: np.ndarray,
    dzdt: np.ndarray,
    pitch_rad: np.ndarray,
    Vbp_m3: np.ndarray,
    density: np.ndarray,
) -> float:
    """Mean-squared error of modelled vs observed vertical velocity.

    All arrays contain only the calibration subset — no masking needed here.
    """
    p = {**base_params}
    for name, val in zip(param_names, x):
        p[name] = val * _SCALE.get(name, 1.0)

    # Re-derive Cd1 if any of its dependencies changed.
    if any(n in param_names for n in ("aw", "eOsborne", "AR", "Cd1_hull")):
        p["Cd1"] = _cd1_from_params(p["aw"], p["eOsborne"], p["AR"], p["Cd1_hull"])

    result = _solve_flight_si(pressure_Pa, dzdt, pitch_rad, Vbp_m3, density, p)
    dzdt_mod = result["vertical_glider_velocity"]

    residual = (dzdt_mod - dzdt) ** 2
    return float(np.nanmean(residual))


def calibrate(ds: xr.Dataset, conf: dict) -> dict:
    """Calibrate flight model parameters against depth-rate.

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset.  Must contain: pressure, pitch, ballast_pumped, density.
    conf : dict
        glide config dict (as returned by config.load_config).

    Returns
    -------
    dict
        All model parameters after calibration (including fixed ones).
    """
    flight_cfg = conf.get("flight", {})
    p = _build_params(flight_cfg)

    required = ["pressure", "pitch", "ballast_pumped", "density", "z"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise ValueError(f"L2 dataset is missing required variables: {missing}")

    # Drop rows with any NaN in the required variables.
    full = ds[required].dropna(dim="time", how="any")

    # Compute dzdt on the full clean timeseries BEFORE subsetting, so that
    # np.gradient sees a continuous sequence with no boundary discontinuities.
    time_s_full = full.time.values.astype("f8") / 1e9
    z_m_full = full.z.values.astype("f8")
    dzdt_full = np.gradient(z_m_full, time_s_full)
    full["dzdt"] = ("time", dzdt_full)

    bounds = flight_cfg.get("bounds", {})
    min_pressure = float(bounds.get("min_pressure", 20.0))
    max_pressure = float(bounds.get("max_pressure", 1000.0))
    time_start = bounds.get("time_start")
    time_end = bounds.get("time_end")

    sub = full
    if time_start is not None or time_end is not None:
        sub = sub.sel(time=slice(time_start, time_end))
    sub = sub.where(
        (sub.pressure > min_pressure) & (sub.pressure < max_pressure)
    ).dropna(dim="time", how="any")

    if sub.time.size < 10:
        raise ValueError(
            f"Too few data points ({sub.time.size}) after applying bounds. "
            "Check flight.bounds in the config."
        )

    # --- extract calibration arrays (already SI or no conversion needed) ------
    pressure_Pa = sub.pressure.values.astype("f8") * 1e4  # dbar → Pa
    dzdt = sub.dzdt.values.astype("f8")  # m s⁻¹, pre-computed
    pitch_rad = np.deg2rad(sub.pitch.values.astype("f8"))  # degrees → radians
    Vbp_m3 = sub.ballast_pumped.values.astype("f8") * 1e-6  # cm³ → m³
    density = sub.density.values.astype("f8")
    # --------------------------------------------------------------------------

    # --- optimisation ---------------------------------------------------------
    calibrate_params = flight_cfg.get("calibrate", ["Vg", "mg", "Cd0", "ah"])
    _log.info("Calibrating parameters: %s", calibrate_params)

    x0 = np.array([p[name] / _SCALE.get(name, 1.0) for name in calibrate_params])

    args = (calibrate_params, p, pressure_Pa, dzdt, pitch_rad, Vbp_m3, density)

    result = fminimize(_cost, x0, args=args, disp=False, xtol=1e-5)

    for name, val in zip(calibrate_params, result):
        p[name] = val * _SCALE.get(name, 1.0)

    # Re-derive Cd1 in case dependent params changed.
    if any(n in calibrate_params for n in ("aw", "eOsborne", "AR", "Cd1_hull")):
        p["Cd1"] = _cd1_from_params(p["aw"], p["eOsborne"], p["AR"], p["Cd1_hull"])

    _log.info(
        "Calibration complete: %s",
        {n: f"{p[n]:.5g}" for n in calibrate_params},
    )
    return p


# ---------------------------------------------------------------------------
# Apply model to full dataset
# ---------------------------------------------------------------------------

_OUTPUT_ATTRS = dict(
    speed_through_water=dict(
        long_name="Speed through water",
        units="m s-1",
        comment="Along-axis incident water speed from steady-state flight model",
        observation_type="calculated",
        valid_min=0.0,
        valid_max=2.0,
    ),
    vertical_glider_velocity=dict(
        long_name="Vertical glider velocity",
        units="m s-1",
        comment=(
            "Vertical component of glider velocity through the water (upward "
            "positive). Negative when diving, positive when climbing."
        ),
        observation_type="calculated",
        valid_min=-1.0,
        valid_max=1.0,
    ),
    vertical_water_velocity=dict(
        long_name="Vertical water velocity",
        units="m s-1",
        comment=(
            "Vertical water velocity estimated as dz/dt minus modelled vertical "
            "glider velocity (upward positive)."
        ),
        observation_type="calculated",
        valid_min=-1.0,
        valid_max=1.0,
    ),
    angle_of_attack=dict(
        long_name="Angle of attack",
        units="degrees",
        comment="Angle of attack of glider relative to oncoming flow",
        observation_type="calculated",
        valid_min=-30.0,
        valid_max=30.0,
    ),
)


def apply_model(ds: xr.Dataset, params: dict) -> xr.Dataset:
    """Solve the flight model over the full dataset and add output variables.

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset.  Must contain: pressure, pitch, ballast_pumped, density.
    params : dict
        Model parameters (e.g. as returned by calibrate()).

    Returns
    -------
    xr.Dataset
        Copy of ``ds`` with flight model variables added on the ``time``
        dimension and calibrated parameters stored as global attributes.
    """
    required = ["pressure", "pitch", "ballast_pumped", "density", "z"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise ValueError(f"L2 dataset is missing required variables: {missing}")

    # --- unit conversion: glide native → SI -----------------------------------
    time_s = ds.time.values.astype("f8") / 1e9
    pressure_Pa = ds.pressure.values.astype("f8") * 1e4  # dbar → Pa
    z_m = ds.z.values.astype("f8")  # m, already SI
    pitch_rad = np.deg2rad(ds.pitch.values.astype("f8"))  # degrees → radians
    Vbp_m3 = ds.ballast_pumped.values.astype("f8") * 1e-6  # cm³ → m³
    density = ds.density.values.astype("f8")
    # Compute dzdt on the full dataset timeseries before any subsetting.
    dzdt = np.gradient(z_m, time_s)
    # --------------------------------------------------------------------------

    result = _solve_flight_si(pressure_Pa, dzdt, pitch_rad, Vbp_m3, density, params)

    out = ds.copy()

    for var_name, values in result.items():
        attrs = _OUTPUT_ATTRS[var_name]
        out[var_name] = (("time",), values.astype("f4"), attrs)

    # Store all model parameters as global attributes.
    _param_keys = [
        "mg",
        "Vg",
        "Cd0",
        "ah",
        "Cd1",
        "aw",
        "S",
        "AR",
        "Omega",
        "eOsborne",
        "Cd1_hull",
        "epsilon",
        "rho0",
    ]
    for key in _param_keys:
        if params.get(key) is not None:
            out.attrs[f"flight_model_{key}"] = params[key]

    return out
