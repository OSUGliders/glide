# Functions applying quality control

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

_log = logging.getLogger(__name__)


def nan_out_of_bounds(y: ArrayLike, y0: float, y1: float) -> NDArray:
    y = np.asarray(y)
    y[(y < y0) | (y > y1)] = np.nan
    return y


def fit_line(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    """Slope and intercept between two points (x0,y0) and (x1,y1)."""
    m = (y1 - y0) / (x1 - x0)
    c = y0 - m * x0
    return m, c


def closely_spaced(t: ArrayLike, dt: float = 1.0) -> NDArray:
    """t is array of points. dt is max gap."""
    if not np.isfinite(t).all():
        raise ValueError("All elements in t must be finite")
    condition = np.diff(t) > dt
    (idx,) = condition.nonzero()
    idx += 1
    start = np.hstack((0, idx))
    end = np.hstack((idx, condition.size))
    idx_bound = np.stack((start, end)).T
    return idx_bound


def extract_surface_gps_groups(
    time: ArrayLike, lon: ArrayLike, lat: ArrayLike, depth: ArrayLike, dt: float = 600
):
    """
    Arguments
    time: timestamp, seconds since 1970-01-01T00:00:00
    lon: Slocum m_gps_lon
    lat: Slocum m_gps_lat
    depth: Slocum m_depth
    dt: Gap finding threshold, seconds

    Returns
    idx_bound: bounding indexes of groups of good surface fixes in original data
    idx_good_updated: indexes of good surface fixes
    """
    time = np.asarray(time)
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    depth = np.asarray(depth)

    good = np.isfinite(lon) & np.isfinite(lat)

    idx_good = np.argwhere(good).flatten()
    t, d = time[idx_good], depth[idx_good]

    d[~np.isfinite(d)] = 0
    d_med = np.median(d)
    _log.debug("Median depth %.2f", d_med)

    if dt is None:
        # The default gap finding threshold
        dt = 3 * np.median(np.diff(t))

    _log.debug("Gap finding threshold %.2f", dt)

    # Indexes bounding closely spaced fixes
    idx_bound_good = closely_spaced(t, dt)

    fix_count = idx_bound_good[:, 1] - idx_bound_good[:, 0]
    _log.debug("Found %i groups of fixes", fix_count.size)
    _log.debug("Count of fixes in groups is %s", fix_count)

    # QC: Remove first fix after long gap
    # TODO: The first fix in the last grouping should not be removed if we follow the IOOS manual
    idx_after_gap = (
        idx_bound_good[:, 0] if fix_count[-1] > 2 else idx_bound_good[:-1, 0]
    )
    _log.debug("%i indexes after gaps to remove", idx_after_gap.size)

    # QC: Remove deep fixes
    idx_depth_bad = np.argwhere(d > d_med + 0.5).flatten()
    _log.debug("%i deep fixes to remove", idx_depth_bad.size)

    # Update good flag
    idx_set_bad = np.hstack((idx_after_gap, idx_depth_bad))
    good_updated = good.copy()
    good_updated[idx_set_bad] = False

    # Recalculate groups
    idx_good_updated = np.argwhere(good_updated).flatten()
    idx_bound_good = closely_spaced(time[idx_good_updated], dt)
    idx_bound = idx_bound_good.copy()
    idx_bound[:, 0] = idx_good_updated[idx_bound_good[:, 0]]
    idx_bound[:, 1] = idx_good_updated[idx_bound_good[:, 1] - 1]
    return idx_bound, idx_good_updated


def changed_elements(x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    x, y = np.asarray(x), np.asarray(y)
    unchanged = np.isclose(x, y)
    both_nan = np.isnan(x) & np.isnan(y)
    changed = np.logical_not(unchanged | both_nan)
    return changed, unchanged


def apply_data_bounds(ds) -> xr.Dataset:
    _log.debug("Flagging out of bounds data")
    for v in ds.data_vars:
        if "qc_" in v:
            continue
        if "valid_min" not in ds[v].attrs or "valid_max" not in ds[v].attrs:
            _log.debug("%s does not have valid_min or valid_max attribute, skipping", v)
            continue

        vmin = ds[v].attrs["valid_min"]
        vmax = ds[v].attrs["valid_max"]

        _log.debug(
            "Removing %s outside %s to %s",
            v,
            vmin,
            vmax,
        )

        original = ds[v].values.copy()

        ds[v] = (ds[v].dims, nan_out_of_bounds(ds[v], vmin, vmax), ds[v].attrs)

        # Update QC
        changed, unchanged = changed_elements(original, ds[v])
        _log.debug("%i elements changed of %i total", changed.sum(), changed.size)
        ds = update_qc_flag(ds, v, 4, changed)  # Outside bounds is bad
        ds = update_qc_flag(ds, v, 2, unchanged)  # Within is probably good

    return ds


def interpolate_missing(ds: xr.Dataset, var_specs: dict) -> xr.Dataset:
    _log.debug("Interpolating missing data")
    for v in ds.data_vars:
        if v not in var_specs:
            continue

        specs = var_specs[v]
        if "interpolate_missing" not in specs:
            continue
        if not specs["interpolate_missing"]:
            continue

        max_gap = specs["max_gap"] if "max_gap" in specs else 60

        _log.debug("Interpolating %s with max gap %s", v, max_gap)

        original = ds[v].values.copy()

        ds[v] = ds[v].interpolate_na(dim="time", max_gap=max_gap)

        # Update QC
        changed, _ = changed_elements(original, ds[v])
        _log.debug("%i elements changed of %i total", changed.sum(), changed.size)
        ds = update_qc_flag(ds, str(v), 9, changed)

    return ds


def time(
    ds: xr.Dataset,
    time_var: str = "time",
    time_bounds: tuple[float, float] = (946684800, 2208988800),
) -> xr.Dataset:
    """Apply time thresholds, remove all NaT data, keep only unique times.
    The default thresholds are timestamps:
    1577808000 = 2000-01-01T00:00:00
    2208960000 = 2040-01-01T00:00:00
    """
    t0, t1 = time_bounds
    _log.debug(
        "Removing times outside %s to %s",
        pd.to_datetime(t0, unit="s"),
        pd.to_datetime(t1, unit="s"),
    )

    dim = ds[time_var].dims[0]
    ds[time_var] = (
        dim,
        nan_out_of_bounds(ds[time_var], t0, t1),
        ds[time_var].attrs,
    )
    good = np.isfinite(ds[time_var])
    _log.debug(
        "%s contains %i good points of %i total",
        time_var,
        good.sum(),
        ds[time_var].size,
    )
    ds = ds.isel({dim: good})

    _, idx = np.unique(ds[time_var], return_index=True)
    _log.debug(
        "%s contains %i unique points of %i total",
        time_var,
        idx.size,
        ds[time_var].size,
    )
    ds = ds.isel({dim: idx})

    # After this we are pretty confident in the times
    ds = update_qc_flag(ds, time_var, 2, np.full(ds[time_var].shape, True))

    return ds


def gps(
    ds: xr.Dataset,
    dt: float = 600,
) -> xr.Dataset:
    """Bad surface GPS fixes are removed. Thresholds are applied.

    Dead reckoning positions are linearly adjusted between surface fixes.

    Should take place after formatting variables.
    """

    # Adjust dead reckoning
    _log.debug("Adjusting dead reckoning")
    time = ds.time.values
    gps_lon = ds.m_gps_lon.values
    gps_lat = ds.m_gps_lat.values
    depth = ds.m_depth.values

    idx_bound, idx_good_gps = extract_surface_gps_groups(
        time, gps_lon, gps_lat, depth, dt
    )
    n_gaps = idx_bound.shape[0] - 1
    _log.debug("Found %i gaps between surface fixes", n_gaps)

    lon = ds.lon.values
    lat = ds.lat.values

    idx_changed = []  # Track changed points for QC

    for i in range(n_gaps):
        # Gaps are between surface fix groups
        in_gap = slice(idx_bound[i, 1], idx_bound[i + 1, 0])

        # Get deadreckoning positions within gap
        lo = lon[in_gap]
        la = lat[in_gap]
        t = time[in_gap]

        good = np.isfinite(lo) & np.isfinite(la)
        if ~good.any():  # Skip empty gaps
            _log.debug("No dead reckoned positions to adjust in gap %i", i)
            continue

        _log.debug("Adjusting %i positions in gap %i", good.sum(), i)

        idx_good = np.argwhere(good).flatten()
        idx_changed.append(idx_good.copy() + in_gap.start)

        # Adjustments to dead reckoning
        m_gps, c_gps = fit_line(
            time[in_gap.start],
            gps_lon[in_gap.start],
            time[in_gap.stop],
            gps_lon[in_gap.stop],
        )
        m_dr, c_dr = fit_line(
            t[idx_good[0]], lo[idx_good[0]], t[idx_good[-1]], lo[idx_good[-1]]
        )

        dmlo, dclo = (m_gps - m_dr), (c_gps - c_dr)
        dlo = dmlo * t + dclo

        _log.debug("lon slope diff: %.2e, intercept: %.2e, in gap %i", dmlo, dclo, i)

        m_gps, c_gps = fit_line(
            time[in_gap.start],
            gps_lat[in_gap.start],
            time[in_gap.stop],
            gps_lat[in_gap.stop],
        )
        m_dr, c_dr = fit_line(
            t[idx_good[0]], la[idx_good[0]], t[idx_good[-1]], la[idx_good[-1]]
        )

        dmla, dcla = (m_gps - m_dr), (c_gps - c_dr)
        dla = dmla * t + dcla

        _log.debug("lat slope diff: %.2e, intercept: %.2e, in gap %i", dmla, dcla, i)

        lon[in_gap] = lo + dlo
        lat[in_gap] = la + dla

    changed = np.full(lon.shape, False)
    changed[np.hstack(idx_changed)] = True

    ds["lon"] = (ds.lon.dims, lon, ds.lon.attrs)
    ds["lat"] = (ds.lat.dims, lat, ds.lat.attrs)

    # QC update
    ds = update_qc_flag(ds, "lon", 2, changed)
    ds = update_qc_flag(ds, "lat", 2, changed)

    # Finally, remove m_gps data that were flagged bad.
    m_gps_lon = np.full_like(gps_lon, np.nan)
    m_gps_lon[idx_good_gps] = gps_lon[idx_good_gps]
    ds["m_gps_lon"] = (ds.m_gps_lon.dims, m_gps_lon, ds.m_gps_lon.attrs)

    m_gps_lat = np.full_like(gps_lat, np.nan)
    m_gps_lat[idx_good_gps] = gps_lat[idx_good_gps]
    ds["m_gps_lat"] = (ds.m_gps_lat.dims, m_gps_lat, ds.m_gps_lat.attrs)

    return ds


def init_qc_var(ds: xr.Dataset, variable: str) -> xr.Dataset:
    _log.debug("Initialising qc flags for %s", variable)

    y = ds[variable]

    qc_attrs = {
        "flag_meanings": "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used not_used interpolated_value missing_value",
        "flag_values": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="b"),
        "long_name": f"{y.attrs['long_name']} Quality Flag",
        "standard_name": f"{y.attrs['standard_name']} status_flag",
        "valid_max": np.int8(9),
        "valid_min": np.int8(0),
    }

    flag = np.zeros_like(y, dtype="b")
    flag[~np.isfinite(y)] = 9

    ds["qc_" + variable] = (y.dims, flag, qc_attrs)

    return ds


def init_dataset_qc(ds: xr.Dataset, var_specs: dict) -> xr.Dataset:
    _log.debug("Initialising all qc flags")

    for v in ds.variables:
        if v not in var_specs:
            continue

        specs = var_specs[v]
        if "track_qc" not in specs:
            continue
        if not specs["track_qc"]:
            continue

        ds = init_qc_var(ds, str(v))

    return ds


def update_qc_flag(
    ds: xr.Dataset, variable: str, flag_value: int, locs: ArrayLike
) -> xr.Dataset:
    """
    flag_values
    0: no_qc_performed
    1: good_data
    2: probably_good_data
    3: bad_data_that_are_potentially_correctable
    4: bad_data
    5: value_changed
    6: not_used
    7: not_used
    8: interpolated_value
    9: missing_value
    """

    locs = np.asarray(locs)

    qcv = "qc_" + variable

    if qcv not in ds.variables:
        return ds

    if np.logical_not(locs).all():
        return ds

    flag = ds[qcv].values
    flag[locs] = np.int8(flag_value)

    ds[qcv] = (ds[qcv].dims, flag, ds[qcv].attrs)

    return ds
