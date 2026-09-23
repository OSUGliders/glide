# Applies lag and thermal mass corrections to the CTD temperature, for the RBR
# legato and the pumped Sea-Bird. Both sensors report `temperature` with at most
# a lag correction applied, and both add `temperature_cell`, the temperature of
# the water in the conductivity cell, which is what salinity is calculated from.

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from . import flight
from .config import _deep_merge

_log = logging.getLogger(__name__)

# Canonical names of the rbrctd suite variables (after format_l1 renaming).
_RBRCTD_VARS = (
    "rbrctd_time",
    "rbrctd_temperature",
    "rbrctd_conductivity",
)

_TS_SENTINEL = 946684800  # 2000-01-01T00:00:00Z

# Presence of this variable selects the Sea-Bird correction. It is a sensor
# timestamp but is deliberately not used as one; see core.yml.
_CTD41CP_VAR = "ctd41cp_time"

# Default correction parameters per sensor, overridable under the `ctd:` section
# of the user config, which mirrors this layout.
DEFAULTS = dict(
    rbrctd=dict(
        temperature_lag=0.9,  # s, manufacturer specified
        thermal_mass=dict(
            bulk=dict(
                alpha_prefactor=0.05,
                alpha_exponent=-0.83,
                tau_prefactor=334.21,
                tau_exponent=0.03,
            ),
            long=dict(
                alpha_prefactor=0.18,
                alpha_exponent=-1.1,
                tau_prefactor=179.0,
                tau_exponent=0.0,
            ),
            short=dict(
                alpha_prefactor=0.23,
                alpha_exponent=-0.82,
                tau_prefactor=27.15,
                tau_exponent=-0.58,
            ),
        ),
    ),
    ctd41cp=dict(
        alpha=0.03,  # fractional amplitude error, Sea-Bird pumped default
        tau=7.0,  # s, response time, Sea-Bird pumped default
    ),
)

_F_N = 0.5  # Hz, Nyquist frequency of the 1 Hz legato sampling
_U_MIN = 0.05  # m s-1, speed floor; the coefficients diverge as U approaches 0
_U_MAX = 0.6  # m s-1, speed ceiling
_U_WINDOW = 31  # samples, rolling mean window applied to the speed estimate
_GAP = 60.0  # s, data gap beyond which the recursive filter state is reset


def _build_native_rbrctd(
    sci: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_s, T, C) on the native rbrctd timestamp grid.

    Drops rows where the native timestamp is missing or below the sentinel,
    sorts ascending, and removes duplicate timestamps.
    """
    t = np.asarray(sci["rbrctd_time"].values, dtype="f8")
    T = np.asarray(sci["rbrctd_temperature"].values, dtype="f8")
    C = np.asarray(sci["rbrctd_conductivity"].values, dtype="f8")

    valid = np.isfinite(t) & (t > _TS_SENTINEL)
    t, T, C = t[valid], T[valid], C[valid]

    order = np.argsort(t, kind="stable")
    t, T, C = t[order], T[order], C[order]

    _, keep = np.unique(t, return_index=True)
    keep.sort()
    return t[keep], T[keep], C[keep]


def _apply_lag(time_s: np.ndarray, T: np.ndarray, lag_s: float) -> np.ndarray:
    """Shift ``T`` earlier in time by ``lag_s`` seconds.

    The sensor reports later than the water it samples, so the "true"
    temperature at time ``t`` equals the value sampled at ``t + lag``.
    Equivalently, interpolate ``T`` from ``time - lag`` back onto ``time``.
    """
    if lag_s == 0.0:
        return T.copy()
    shifted = time_s - lag_s
    return np.interp(time_s, shifted, T)


def _filter_coefficients(U: np.ndarray, coeffs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recursive filter coefficients (a, b) for the thermal mass correction.

    ``alpha`` (fractional amplitude error) and ``tau`` (response time) follow
    RBR's power laws in glider speed, which are defined for speed in cm s-1
    while ``U`` is in m s-1 - hence the factor of 100.
    """
    alpha = coeffs["alpha_prefactor"] * (100 * U) ** coeffs["alpha_exponent"]
    tau = coeffs["tau_prefactor"] * (100 * U) ** coeffs["tau_exponent"]
    gain = 4 * _F_N * tau / (1 + 4 * _F_N * tau)
    return alpha * gain, 1 - 2 * gain


def _thermal_mass(
    T: np.ndarray, U: np.ndarray, time_s: np.ndarray, coeffs: dict
) -> np.ndarray:
    """Thermal mass correction of Morison et al. (1994) as a recursive filter.

    The filter state is reset across data gaps longer than ``_GAP`` seconds.
    """
    a, b = _filter_coefficients(U, coeffs)
    T_TM = np.zeros_like(T)
    for i in range(1, T.size):
        if time_s[i] - time_s[i - 1] > _GAP:
            continue
        T_TM[i] = -b[i] * T_TM[i - 1] + a[i] * (T[i] - T[i - 1])
    return T_TM


def _correct_thermal_mass(
    T: np.ndarray, U: np.ndarray, time_s: np.ndarray, coeffs: dict
) -> np.ndarray:
    """Correct ``T`` for the thermal mass of the sensor.

    The bulk correction removes the thermistor housing response; the long and
    short corrections then remove the slow and fast heat exchange with the
    sensor body and stem respectively.
    """
    T_bulk = T + _thermal_mass(T, U, time_s, coeffs["bulk"])
    return (
        T_bulk
        - _thermal_mass(T_bulk, U, time_s, coeffs["long"])
        - _thermal_mass(T_bulk, U, time_s, coeffs["short"])
    )


def _cell_thermal_mass(
    T: np.ndarray, time_s: np.ndarray, alpha: float, tau: float
) -> np.ndarray:
    """Return the conductivity cell temperature for constant ``alpha`` and ``tau``.

    The pumped Sea-Bird flushes its cell at a rate set by the pump rather than by
    the glider, so unlike the legato its coefficients do not depend on speed. The
    correction is the Morison et al. (1994) recursive filter, evaluated at the
    Nyquist frequency of each sampling interval, and is subtracted: the water in
    the cell is warmer than ambient while the glider descends into colder water.
    """
    dt = np.diff(time_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.0 / (2.0 * dt)
    a = 4 * f * alpha * tau / (1 + 4 * f * tau)
    b = 1 - 2 * a / alpha

    corr = np.zeros_like(T)
    for i in range(1, T.size):
        if not (0 < dt[i - 1] <= _GAP):
            continue
        corr[i] = -b[i - 1] * corr[i - 1] + a[i - 1] * (T[i] - T[i - 1])
    return T - corr


def _add_temperature_cell(
    sci: xr.Dataset, values: np.ndarray, config: dict
) -> xr.Dataset:
    """Add ``temperature_cell`` to ``sci`` with its CF attrs from the config."""
    specs = config.get("variables", {}).get("temperature_cell", {})
    sci["temperature_cell"] = (
        sci["temperature"].dims,
        values,
        specs.get("CF", {}),
    )
    return sci


def _interp_finite(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray | None:
    """Interpolate ``y(x)`` onto ``xi``, ignoring non-finite source values."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    return np.interp(xi, x[ok], y[ok])


def _bound_speed(U: np.ndarray) -> np.ndarray:
    """Clip, smooth and gap-fill a speed estimate for the thermal mass filter.

    The filter coefficients are ill-behaved near a profile apex, where the speed
    estimate passes through zero, and a single non-finite speed would poison the
    recursion for every later sample.
    """
    U = (
        pd.Series(np.clip(U, _U_MIN, _U_MAX))
        .rolling(_U_WINDOW, center=True, min_periods=_U_WINDOW // 2)
        .mean()
        .to_numpy()
    )
    return np.where(np.isfinite(U), U, _U_MIN)


def _native_speed(
    sci: xr.Dataset, flt: xr.Dataset | None, t_native: np.ndarray
) -> np.ndarray | None:
    """Glider speed on the native CTD grid, or None if it cannot be estimated."""
    if flt is None or "pitch" not in flt.variables or "pressure" not in sci.variables:
        _log.warning(
            "No flight pitch or science pressure; "
            "skipping the CTD thermal mass correction"
        )
        return None

    pressure = _interp_finite(
        _time_as_seconds(sci["time"]), sci["pressure"].values, t_native
    )
    pitch = _interp_finite(_time_as_seconds(flt["time"]), flt["pitch"].values, t_native)
    if pressure is None or pitch is None:
        _log.warning(
            "Insufficient pressure or pitch data; "
            "skipping the CTD thermal mass correction"
        )
        return None

    lat = np.nanmedian(flt["lat"].values) if "lat" in flt.variables else np.nan
    U = flight.estimate_speed(
        t_native,
        pressure,
        np.deg2rad(pitch),
        float(lat) if np.isfinite(lat) else 0.0,
    )
    return _bound_speed(U)


def _overwrite(
    sci: xr.Dataset, name: str, values: np.ndarray, comment: str
) -> xr.Dataset:
    """Overwrite ``sci[name]`` while preserving its CF attrs and appending a comment."""
    if name not in sci.variables:
        _log.warning("Cannot overwrite %s: not present in science dataset", name)
        return sci
    attrs = dict(sci[name].attrs)
    existing = attrs.get("comment", "")
    attrs["comment"] = f"{existing}; {comment}".lstrip("; ")
    sci[name] = (sci[name].dims, values, attrs)
    return sci


def _time_as_seconds(t: xr.DataArray) -> np.ndarray:
    """Return ``t`` as float64 posix seconds regardless of dtype."""
    vals = t.values
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]").astype("f8") / 1e9
    return np.asarray(vals, dtype="f8")


def _correct_rbrctd(
    sci: xr.Dataset, cfg: dict, config: dict, flt: xr.Dataset | None
) -> xr.Dataset:
    """Lag and thermal mass correction for the RBR legato.

    Shifts temperature earlier in time on the native ``rbrctd_time`` grid by
    ``temperature_lag`` seconds, interpolates the lag-shifted temperature and the
    unadjusted conductivity back onto ``sci.time``, then adds the additionally
    thermal mass corrected ``temperature_cell``. The thermal mass stages need an
    estimate of glider speed, so they are skipped without flight data.
    """
    lag_s = float(cfg["temperature_lag"])

    t_native, T, C = _build_native_rbrctd(sci)
    if t_native.size < 2:
        _log.warning("rbrctd has fewer than 2 valid samples; skipping correction")
        return sci

    T_lag = _apply_lag(t_native, T, lag_s)

    sci_t = _time_as_seconds(sci["time"])
    T_on_sci = np.interp(sci_t, t_native, T_lag, left=np.nan, right=np.nan)
    C_on_sci = np.interp(sci_t, t_native, C, left=np.nan, right=np.nan)

    sci = _overwrite(
        sci,
        "temperature",
        T_on_sci,
        f"lag-corrected (lag={lag_s}s) via ctd.correct_ctd",
    )
    sci = _overwrite(
        sci,
        "conductivity",
        C_on_sci,
        "interpolated from rbrctd native grid via ctd.correct_ctd",
    )

    U = _native_speed(sci, flt, t_native)
    if U is None:
        return sci

    T_cell = _correct_thermal_mass(T_lag, U, t_native, cfg["thermal_mass"])
    return _add_temperature_cell(
        sci, np.interp(sci_t, t_native, T_cell, left=np.nan, right=np.nan), config
    )


def _correct_ctd41cp(sci: xr.Dataset, cfg: dict, config: dict) -> xr.Dataset:
    """Cell thermal mass correction for the pumped Sea-Bird CTD.

    The Sea-Bird reports through the canonical ``temperature`` variable on the
    science time grid, so unlike the legato there is no native grid to rebuild
    and no lag to remove: only ``temperature_cell`` is added.

    The filter runs over the samples the sensor actually reported, which
    ``ctd41cp_time`` identifies: elsewhere the sensor fills temperature with
    exact zeros, and those pass the QC bounds check but would enter the
    recursion as multi-degree steps. The result is then interpolated onto the
    full science time grid.
    """
    t = _time_as_seconds(sci["time"])
    T = np.asarray(sci["temperature"].values, dtype="f8")
    reported = np.asarray(sci[_CTD41CP_VAR].values, dtype="f8") > _TS_SENTINEL
    ok = reported & np.isfinite(T) & np.isfinite(t)
    if ok.sum() < 2:
        _log.warning("ctd41cp has fewer than 2 valid samples; skipping correction")
        return sci

    T_cell = _cell_thermal_mass(T[ok], t[ok], float(cfg["alpha"]), float(cfg["tau"]))
    return _add_temperature_cell(
        sci, np.interp(t, t[ok], T_cell, left=np.nan, right=np.nan), config
    )


def correct_ctd(
    sci: xr.Dataset, config: dict, flt: xr.Dataset | None = None
) -> xr.Dataset:
    """Apply CTD lag and thermal mass corrections to the science dataset.

    The sensor is identified from the variables present: the RBR legato by its
    ``rbrctd_*`` suite, the pumped Sea-Bird by ``ctd41cp_time``. Both leave the
    reported ``conductivity`` unadjusted and add ``temperature_cell``, which is
    what ``process_l1.calculate_thermodynamics`` calculates salinity from.

    Parameters
    ----------
    sci : xr.Dataset
        Formatted science dataset, before merging with flight.
    config : dict
        Configuration; correction parameters are read from the `ctd:` section.
    flt : xr.Dataset, optional
        Formatted flight dataset, used by the legato correction for pitch and
        latitude.

    Returns
    -------
    xr.Dataset
        ``sci`` unchanged when no recognised CTD is present.
    """
    cfg = _deep_merge(DEFAULTS, config.get("ctd") or {})

    if all(v in sci.variables for v in _RBRCTD_VARS):
        return _correct_rbrctd(sci, cfg["rbrctd"], config, flt)

    if _CTD41CP_VAR in sci.variables:
        return _correct_ctd41cp(sci, cfg["ctd41cp"], config)

    _log.debug("No recognised CTD variables in science dataset; skipping correction")
    return sci
