# Steady-state glider flight model following Merckelbach et al. 2010 & 2019.

import logging

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, least_squares

from .config import _load_core
from .convert import cm3_to_m3, dbar_to_Pa, deg_to_rad

_log = logging.getLogger(__name__)


# Default model parameters
DEFAULTS = dict(
    rho0=1025.0,  # reference density (kg m-3)
    mg=70.0,  # glider mass (kg)
    Vg=0.090,  # glider volume (m3)
    Cd0=0.15,  # parasitic drag coefficient (-)
    Cd1=10.0,  # induced drag coefficient (s-2); derived from aw if not set
    Cd1_hull=9.7,  # hull induced-drag coefficient
    ah=3.8,  # hull lift coefficient slope rad-1)
    aw=None,  # wing lift coefficient slope (rad-1); derived from aspect_ratio/Omega
    reference_area=0.10,  # reference area (m-2)
    aspect_ratio=7.0,  # aspect ratio
    sweep_angle=0.7505,  # sweep angle (rad, approx 43 degrees)
    osborne_efficiency=0.8,  # Osborne efficiency factor
    hull_compressibility=5e-10,  # hull compressibility (Pa-1)
)

_G = 9.81  # gravity m s-2

# Physical bounds for calibratable parameters.
BOUNDS: dict[str, tuple[float, float]] = dict(
    mg=(40.0, 150.0),  # glider mass (kg)
    Vg=(0.005, 0.2),  # glider volume (m3)
    Cd0=(0.0, 2.0),  # parasitic drag coefficient
    ah=(0.0, 10.0),  # hull lift slope (rad-1)
    Cd1=(0.0, 100.0),  # induced drag coefficient
    aw=(0.0, 30.0),  # wing lift slope (rad-1)
)


def _aw_from_geometry(aspect_ratio: float, sweep_angle: float) -> float:
    """Lift-curve slope from wing aspect ratio and sweep angle."""
    return (
        2
        * np.pi
        * aspect_ratio
        / (2 + np.sqrt(aspect_ratio**2 * (1 + np.tan(sweep_angle) ** 2) + 4))
    )


def _cd1_from_params(
    aw: float, osborne_efficiency: float, aspect_ratio: float, Cd1_hull: float
) -> float:
    """Induced drag coefficient from wing and hull contributions."""
    Cd1_wing = aw**2 / (np.pi * osborne_efficiency * aspect_ratio)
    return Cd1_wing + Cd1_hull


def _build_params(cfg: dict) -> dict:
    """Merge user config over defaults, derive aw / Cd1 if not explicitly set."""
    p = {**DEFAULTS, **cfg}

    if p["aw"] is None:
        p["aw"] = _aw_from_geometry(p["aspect_ratio"], p["sweep_angle"])
        _log.debug("Derived aw = %.4f from aspect_ratio and sweep_angle", p["aw"])

    # Cd1 is derived unless the caller overrides it directly.
    if "Cd1" not in cfg:
        p["Cd1"] = _cd1_from_params(
            p["aw"], p["osborne_efficiency"], p["aspect_ratio"], p["Cd1_hull"]
        )
        _log.debug("Derived Cd1 = %.4f", p["Cd1"])

    return p


def _to_si(ds: xr.Dataset) -> dict[str, np.ndarray]:
    """Extract flight model inputs from ds and convert to SI units.

    If ``dzdt`` is already present in ds (pre-computed on a larger timeseries),
    it is used directly; otherwise it is computed via np.gradient on z and time.

    Returns a dict with keys: pressure [Pa], dzdt [m s-1], pitch [rad],
    Vbp [m3], density [kg m-3].
    """
    time_s = ds.time.values.astype("f8") / 1e9
    dzdt = (
        ds.dzdt.values.astype("f8")
        if "dzdt" in ds
        else np.gradient(ds.z.values.astype("f8"), time_s)
    )
    return dict(
        pressure=dbar_to_Pa(ds.pressure.values.astype("f8")),
        dzdt=dzdt,
        pitch=deg_to_rad(ds.pitch.values.astype("f8")),
        Vbp=cm3_to_m3(ds.ballast_pumped.values.astype("f8")),
        density=ds.density.values.astype("f8"),
    )


def _compute_buoyancy(
    pressure: np.ndarray,
    rho: np.ndarray,
    Vbp: np.ndarray,
    Vg: float,
    hull_compressibility: float,
    mg: float,
) -> tuple[np.ndarray, float]:
    """Return buoyancy force FB (N) and gravity force Fg (N).

    pressure : Pa
    Vbp : m3

    Merckelbach et al. 2019 equation 3 & 4.
    """
    FB = _G * rho * (Vg * (1.0 - hull_compressibility * pressure) + Vbp)
    Fg = mg * _G
    return FB, Fg


def _aoa_equation(
    alpha: float, pitch: float, Cd0: float, Cd1: float, ah: float, aw: float
) -> float:
    """Implicit equation whose zero is the angle of attack.

    pitch : rad

    Merckelbach et al. 2019 equation 9.
    """
    return (Cd0 + Cd1 * alpha**2) / ((ah + aw) * np.tan(alpha + pitch)) - alpha


def _solve_aoa(
    pitch: np.ndarray, Cd0: float, Cd1: float, ah: float, aw: float
) -> np.ndarray:
    """Solve for angle of attack at each pitch sample.

    pitch : rad

    Builds an interpolating table over the pitch range for efficiency, then
    applies it.  Pitches whose magnitude is below ~5 degrees receive aoa = 0.
    """
    pitch_abs = np.abs(pitch)
    pitch_grid = np.linspace(5 * np.pi / 180, 60 * np.pi / 180, 120)

    aoa_grid = np.zeros_like(pitch_grid)
    for i, _pitch in enumerate(pitch_grid):
        sol = fsolve(
            _aoa_equation,
            _pitch / 50.0,
            args=(_pitch, Cd0, Cd1, ah, aw),
            full_output=True,
        )
        aoa_grid[i] = sol[0][0]

    ifun = interp1d(pitch_grid, aoa_grid, bounds_error=False, fill_value=0.0)

    aoa = np.where(
        pitch_abs >= pitch_grid[0],
        ifun(pitch_abs) * np.sign(pitch),
        0.0,
    )
    return aoa


def _compute_speed(
    FB: np.ndarray,
    Fg: float,
    rho: np.ndarray,
    alpha: np.ndarray,
    pitch: np.ndarray,
    reference_area: float,
    Cd0: float,
    Cd1: float,
) -> np.ndarray:
    """Incident water speed U (m s-1) from force balance.

    pitch : rad

    Merckelbach et al. 2019 equation 8.
    """
    numer = 2.0 * (FB - Fg) * np.sin(pitch + alpha)
    denom = rho * reference_area * (Cd0 + Cd1 * alpha**2)
    U2 = np.where(denom != 0, numer / denom, 0.0)
    U2 = np.maximum(U2, 0.0)
    return np.sqrt(U2)


def _solve_flight(
    pressure: np.ndarray,
    dzdt: np.ndarray,
    pitch: np.ndarray,
    Vbp: np.ndarray,
    density: np.ndarray,
    p: dict,
) -> dict:
    """Core steady-state flight model. All inputs in SI units.

    Parameters
    ----------
    pressure : Pa
    dzdt : m s-1, observed vertical velocity (upward positive)
    pitch : rad
    Vbp : m3, ballast volume change
    density : kg m-3, in-situ seawater density

    Returns
    -------
    dict with keys: speed_through_water [m s-1], vertical_glider_velocity
    [m s-1], vertical_water_velocity [m s-1], angle_of_attack [degrees].
    """
    FB, Fg = _compute_buoyancy(
        pressure, density, Vbp, p["Vg"], p["hull_compressibility"], p["mg"]
    )
    alpha = _solve_aoa(pitch, p["Cd0"], p["Cd1"], p["ah"], p["aw"])
    U = _compute_speed(
        FB, Fg, density, alpha, pitch, p["reference_area"], p["Cd0"], p["Cd1"]
    )

    wg = U * np.sin(pitch + alpha)
    ww = dzdt - wg  # vertical water velocity

    return dict(
        speed_through_water=U,
        vertical_glider_velocity=wg,
        vertical_water_velocity=ww,
        angle_of_attack=np.rad2deg(alpha),
    )


def _residuals(
    x: np.ndarray,
    param_names: list[str],
    base_params: dict,
    pressure: np.ndarray,
    dzdt: np.ndarray,
    pitch: np.ndarray,
    Vbp: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Residual vector (modelled minus observed vertical velocity) for least_squares.

    pressure : Pa; pitch : rad; Vbp : m3
    """
    p = {**base_params}
    for name, val in zip(param_names, x):
        p[name] = val

    # Re-derive Cd1 if any of its dependencies changed.
    if any(
        n in param_names
        for n in ("aw", "osborne_efficiency", "aspect_ratio", "Cd1_hull")
    ):
        p["Cd1"] = _cd1_from_params(
            p["aw"], p["osborne_efficiency"], p["aspect_ratio"], p["Cd1_hull"]
        )

    result = _solve_flight(pressure, dzdt, pitch, Vbp, density, p)
    return result["vertical_glider_velocity"] - dzdt


def calibrate(ds: xr.Dataset, conf: dict) -> dict:
    """Calibrate flight model parameters against depth-rate.

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset.  Must contain: pressure, pitch, ballast_pumped, density, z.
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

    # Drop rows with any NaN, then compute dzdt on the clean contiguous timeseries
    # before any pressure/time subsetting (so edge samples aren't lost).
    full = ds[required].dropna(dim="time", how="any")
    full["dzdt"] = ("time", _to_si(full)["dzdt"])

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

    if sub.time.size < 100:
        raise ValueError(
            f"Fewer than 100 data points ({sub.time.size}) after applying bounds. "
            "Check flight.bounds in the config."
        )

    # _to_si picks up the pre-computed dzdt stored in sub["dzdt"]
    si = _to_si(sub)

    calibrate_params = flight_cfg.get("calibrate", ["Vg", "mg", "Cd0", "ah"])
    _log.info("Calibrating parameters: %s", calibrate_params)

    x0 = np.array([p[name] for name in calibrate_params])
    lb = np.array([BOUNDS.get(name, (-np.inf, np.inf))[0] for name in calibrate_params])
    ub = np.array([BOUNDS.get(name, (-np.inf, np.inf))[1] for name in calibrate_params])

    args = (
        calibrate_params,
        p,
        si["pressure"],
        si["dzdt"],
        si["pitch"],
        si["Vbp"],
        si["density"],
    )

    result = least_squares(
        _residuals, x0, args=args, bounds=(lb, ub), x_scale="jac", method="trf"
    )

    for name, val in zip(calibrate_params, result.x):
        p[name] = val

    # Re-derive Cd1 in case dependent params changed.
    if any(
        n in calibrate_params
        for n in ("aw", "osborne_efficiency", "aspect_ratio", "Cd1_hull")
    ):
        p["Cd1"] = _cd1_from_params(
            p["aw"], p["osborne_efficiency"], p["aspect_ratio"], p["Cd1_hull"]
        )

    _log.info(
        "Calibration complete: %s",
        {n: f"{p[n]:.5g}" for n in calibrate_params},
    )
    return p


def apply_model(ds: xr.Dataset, params: dict) -> xr.Dataset:
    """Solve the flight model over the full dataset and add output variables.

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset.  Must contain: pressure, pitch, ballast_pumped, density, z.
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

    si = _to_si(ds)
    result = _solve_flight(
        si["pressure"], si["dzdt"], si["pitch"], si["Vbp"], si["density"], params
    )

    out = ds.copy()

    flight_vars = _load_core()["suites"]["flight_model"]
    for var_name, values in result.items():
        attrs = flight_vars[var_name]["CF"]
        out[var_name] = (("time",), values.astype("f4"), attrs)

    # Store all model parameters as global attributes.
    _param_keys = [
        "mg",
        "Vg",
        "Cd0",
        "ah",
        "Cd1",
        "aw",
        "reference_area",
        "aspect_ratio",
        "sweep_angle",
        "osborne_efficiency",
        "Cd1_hull",
        "hull_compressibility",
        "rho0",
    ]
    for key in _param_keys:
        if params.get(key) is not None:
            out.attrs[f"flight_model_{key}"] = params[key]

    return out
