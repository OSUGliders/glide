# Applies sensor-specific lag corrections to the CTD temperature, then
# re-interpolates temperature and conductivity from the CTD-native timestamp
# grid back onto the science time grid.

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

_log = logging.getLogger(__name__)

# Canonical names of the rbrctd suite variables (after format_l1 renaming).
_RBRCTD_VARS = (
    "rbrctd_time",
    "rbrctd_temperature",
    "rbrctd_conductivity",
)

_TS_SENTINEL = 946684800  # 2000-01-01T00:00:00Z


def correct_ctd(sci: xr.Dataset, config: dict) -> xr.Dataset:
    """Apply CTD lag correction to the science dataset.

    For the RBR Concerto: shift temperature earlier in time on the native
    ``rbrctd_time`` grid by ``ctd.rbrctd.temperature_lag`` seconds, then
    interpolate the lag-shifted temperature and the unadjusted conductivity
    back onto ``sci.time``, overwriting ``temperature`` and ``conductivity``.

    Returns ``sci`` unchanged when the rbrctd variables are not present.
    Other CTDs are not yet supported.
    """
    if not all(v in sci.variables for v in _RBRCTD_VARS):
        _log.debug(
            "rbrctd variables not present in science dataset; skipping CTD correction"
        )
        return sci

    lag_s = float(
        ((config.get("ctd") or {}).get("rbrctd") or {}).get("temperature_lag", 0.9)
    )

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
    return sci


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
