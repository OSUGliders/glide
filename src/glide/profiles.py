import logging

import numpy as np
import xarray as xr
from profinder import find_profiles

_log = logging.getLogger(__name__)


def get_profiles(
    ds: xr.Dataset,
    shallowest_profile: float,
    min_surface_time: float = 180.0,
) -> xr.Dataset:
    """Identify dive and climb profiles from a pressure time series.

    Parameters
    ----------
    ds : xr.Dataset
        Merged L2 dataset with 'pressure' and 'time' variables.
    shallowest_profile : float
        Minimum peak pressure (dbar) for a profile to be recognised.
    min_surface_time : float
        Minimum time (seconds) between consecutive dive apexes.  Used to
        compute sample-count thresholds so the detector adapts to the data
        sampling rate.  A value of ~180 s works for most Slocum deployments.
    """
    raw_diff = np.diff(ds.time.values)
    if np.issubdtype(raw_diff.dtype, np.timedelta64):
        dt_s = float(np.nanmedian(raw_diff.astype("timedelta64[s]").astype("f8")))
    else:
        dt_s = float(np.nanmedian(raw_diff))
    fs = 1.0 / dt_s

    peaks_kwargs = {
        "height": shallowest_profile,
        "prominence": shallowest_profile,
        "distance": max(4, round(min_surface_time * fs)),
        "width": max(2, round(20 * fs)),  # ≥20 s half-width detects ~10 m dives
    }
    troughs_kwargs = {
        "prominence": 2,
        "distance": max(2, round(30 * fs)),
        "width": max(1, round(5 * fs)),
    }

    _log.debug(
        "fs=%.3f Hz; peaks_kwargs=%s; troughs_kwargs=%s",
        fs,
        peaks_kwargs,
        troughs_kwargs,
    )

    profiles = find_profiles(
        ds.pressure.values,
        peaks_kwargs=peaks_kwargs,
        troughs_kwargs=troughs_kwargs,
        missing="drop",
    )

    _log.debug("Found %i profiles", len(profiles))

    n = ds.time.size
    dive_id = np.full(n, -1, dtype="i4")
    climb_id = np.full(n, -1, dtype="i4")
    profile_id = np.full(n, -1, dtype="i4")
    state = np.full(n, -1, dtype="b")

    dive_counter = 1
    climb_counter = 1
    profile_counter = 1

    for prof in profiles:
        dive_start, dive_end, climb_start, climb_end = prof

        dive_id[dive_start:dive_end] = dive_counter
        profile_id[dive_start:dive_end] = profile_counter
        state[dive_start:dive_end] = 1
        dive_counter += 1
        profile_counter += 1

        climb_id[climb_start:climb_end] = climb_counter
        profile_id[climb_start:climb_end] = profile_counter
        state[climb_start:climb_end] = 2
        climb_counter += 1
        profile_counter += 1

    ds["dive_id"] = ("time", dive_id, dict(_FillValue=np.int32(-1)))
    ds["climb_id"] = ("time", climb_id, dict(_FillValue=np.int32(-1)))
    ds["profile_id"] = ("time", profile_id, dict(_FillValue=np.int32(-1)))
    ds["state"] = (
        "time",
        state,
        dict(
            long_name="Glider state",
            flag_values=np.array([-1, 0, 1, 2], "b"),
            flag_meanings="unknown surface dive climb",
            valid_max=np.int8(2),
            valid_min=np.int8(-1),
        ),
    )

    return ds


def assign_surface_state(
    ds: xr.Dataset,
    flt: xr.Dataset | None = None,
    dt: float = 300.0,
    surface_pressure: float = 2.0,
) -> xr.Dataset:
    """Assign surface state (0) to unknown points near GPS fixes.

    A point with state == -1 (unknown) is marked as surface (0) only if it is
    within `dt` seconds of a valid GPS fix AND shallower than `surface_pressure`
    dbar.  The pressure gate prevents the broad GPS time window from reaching
    into adjacent dives.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with state variable from get_profiles.
    flt : xr.Dataset, optional
        Raw flight data containing m_gps_lat/lon with valid GPS fix times.
    dt : float
        Time threshold in seconds for matching to GPS fixes (default 300).
    surface_pressure : float
        Maximum pressure (dbar) for a point to be considered at the surface
        (default 2.0).
    """
    if flt is None or "m_gps_lat" not in flt:
        _log.warning("No flight data with GPS, cannot assign surface state")
        return ds

    gps_valid = np.isfinite(flt.m_gps_lat.values)
    if not gps_valid.any():
        _log.warning("No valid GPS fixes found")
        return ds

    gps_times = np.sort(flt.m_present_time.values[gps_valid])

    state = ds.state.values.copy()
    time_l2 = ds.time.values
    unknown_mask = state == -1

    if unknown_mask.any():
        unknown_times = time_l2[unknown_mask]
        pos = np.searchsorted(gps_times, unknown_times)
        dist_left = np.where(
            pos > 0,
            unknown_times - gps_times[np.maximum(pos - 1, 0)],
            np.inf,
        )
        dist_right = np.where(
            pos < len(gps_times),
            gps_times[np.minimum(pos, len(gps_times) - 1)] - unknown_times,
            np.inf,
        )
        is_near = np.minimum(dist_left, dist_right) <= dt
        is_shallow = np.isfinite(ds.pressure.values[unknown_mask]) & (
            ds.pressure.values[unknown_mask] < surface_pressure
        )
        is_near = is_near & is_shallow

        state[unknown_mask] = np.where(is_near, np.int8(0), state[unknown_mask])
        _log.debug("Assigned %d points to surface state", int(is_near.sum()))

    ds["state"] = ("time", state, ds.state.attrs)
    return ds


def _extract_velocity_data(
    flt: xr.Dataset | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract valid velocity data from flight dataset.

    Returns
    -------
    tuple of (times, u, v) arrays with only valid velocity points,
    or (None, None, None) if no valid data.
    """
    if flt is None:
        _log.warning("No flight data provided, velocity will be NaN")
        return None, None, None

    if "m_water_vx" not in flt or "m_water_vy" not in flt:
        _log.warning("No m_water_vx/vy in flight data, velocity will be NaN")
        return None, None, None

    time_flt = flt.m_present_time.values
    u_flt = flt.m_water_vx.values
    v_flt = flt.m_water_vy.values

    vel_valid = np.isfinite(u_flt) & np.isfinite(v_flt)
    if not vel_valid.any():
        _log.warning("No valid velocity data in flight data")
        return None, None, None

    times = time_flt[vel_valid]
    u_vals = u_flt[vel_valid]
    v_vals = v_flt[vel_valid]

    order = np.argsort(times)
    times = times[order]
    u_vals = u_vals[order]
    v_vals = v_vals[order]

    # Multiple reports per surfacing are refined estimates; take the last (best)
    # from each cluster. A gap > 600 s marks a new dive/surface cycle.
    new_surfacing = np.concatenate(([True], np.diff(times) > 600))
    ends = np.concatenate((np.where(new_surfacing)[0][1:], [len(times)])) - 1

    _log.debug(
        "Clustered %d velocity reports into %d surfacing events",
        len(times),
        len(ends),
    )

    return times[ends], u_vals[ends], v_vals[ends]


def add_velocity(
    ds: xr.Dataset,
    config: dict,
    flt: xr.Dataset | None = None,
) -> xr.Dataset:
    """Add depth-averaged velocity variables to L2 dataset.

    Groups profiles by velocity reports from the raw flight data. Each velocity
    report (m_water_vx/vy) marks a surfacing. All profiles between consecutive
    velocity reports belong to one group and share that velocity estimate.
    An additional group is created for any trailing profiles with no velocity
    report yet (for later backfill).

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset with dive_id and climb_id from get_profiles.
    config : dict
        Configuration with variable specifications.
    flt : xr.Dataset, optional
        Raw flight data containing m_water_vx/vy.
    """
    specs = config["variables"]

    def _empty_velocity():
        ds["time_uv"] = (("time_uv",), [np.nan], specs["time_uv"]["CF"])
        ds["lat_uv"] = (("time_uv",), [np.nan], specs["lat_uv"]["CF"])
        ds["lon_uv"] = (("time_uv",), [np.nan], specs["lon_uv"]["CF"])
        ds["u"] = (("time_uv",), [np.nan], specs["u"]["CF"])
        ds["v"] = (("time_uv",), [np.nan], specs["v"]["CF"])
        return ds

    if "dive_id" not in ds or "climb_id" not in ds:
        _log.warning("No dive/climb IDs in dataset - run get_profiles first")
        return _empty_velocity()

    time_l2 = ds.time.values
    lat_l2 = ds.lat.values
    lon_l2 = ds.lon.values
    is_profile = (ds.dive_id.values >= 0) | (ds.climb_id.values >= 0)

    if not is_profile.any():
        _log.warning("No profiles found in dataset")
        return _empty_velocity()

    vel_times, vel_u, vel_v = _extract_velocity_data(flt)

    groups = []

    if (
        vel_times is not None
        and vel_u is not None
        and vel_v is not None
        and len(vel_times) > 0
    ):
        for i, t_vel in enumerate(vel_times):
            t_start = vel_times[i - 1] if i > 0 else -np.inf
            mask = (time_l2 > t_start) & (time_l2 <= t_vel) & is_profile
            if mask.any():
                groups.append((mask, vel_u[i], vel_v[i]))

        mask = (time_l2 > vel_times[-1]) & is_profile
        if mask.any():
            groups.append((mask, np.nan, np.nan))
    else:
        if is_profile.any():
            groups.append((is_profile, np.nan, np.nan))

    if not groups:
        return _empty_velocity()

    n_groups = len(groups)
    time_uv = np.full(n_groups, np.nan)
    lat_uv = np.full(n_groups, np.nan)
    lon_uv = np.full(n_groups, np.nan)
    u = np.full(n_groups, np.nan)
    v = np.full(n_groups, np.nan)

    for i, (mask, u_val, v_val) in enumerate(groups):
        time_uv[i] = np.nanmean(time_l2[mask])
        lat_uv[i] = np.nanmean(lat_l2[mask])
        lon_uv[i] = np.nanmean(lon_l2[mask])
        u[i] = u_val
        v[i] = v_val
        if np.isfinite(u_val):
            _log.debug("Group %d: assigned u=%.4f, v=%.4f", i, u_val, v_val)

    ds["time_uv"] = (("time_uv",), time_uv, specs["time_uv"]["CF"])
    ds["lat_uv"] = (("time_uv",), lat_uv, specs["lat_uv"]["CF"])
    ds["lon_uv"] = (("time_uv",), lon_uv, specs["lon_uv"]["CF"])
    ds["u"] = (("time_uv",), u, specs["u"]["CF"])
    ds["v"] = (("time_uv",), v, specs["v"]["CF"])

    _log.info(
        "Added velocity for %d groups, %d with valid data",
        n_groups,
        int(np.isfinite(u).sum()),
    )
    return ds


def add_gps_fixes(ds: xr.Dataset, flt: xr.Dataset, config: dict) -> xr.Dataset:
    """Add surface GPS fixes on a separate time_gps dimension.

    Valid (non-NaN) fixes from the post-QC flight dataset are placed on a
    time_gps coordinate, independent of the science time vector.

    Any variable in the config with ``companion_dim: time_gps`` is also placed
    on this dimension at the valid fix indices, provided the variable is present
    in the flight dataset. The validity mask is defined by the ``anchor``
    variables listed on the ``time_gps`` config entry.
    """
    specs = config["variables"]
    time_gps_spec = specs.get("time_gps", {})
    anchor_vars = time_gps_spec.get("anchor", ["lat_gps", "lon_gps"])

    anchor_present = [v for v in anchor_vars if v in flt]
    if not anchor_present:
        _log.debug(
            "No anchor variables for time_gps in flight dataset, skipping GPS fixes"
        )
        return ds

    time_vals = flt.time.values
    valid = np.ones(len(time_vals), dtype=bool)
    for v in anchor_present:
        valid &= np.isfinite(flt[v].values)

    n_valid = int(valid.sum())
    if n_valid == 0:
        _log.debug("No valid GPS fixes found")
        return ds

    _log.debug("Adding %d surface GPS fixes on time_gps dimension", n_valid)
    ds["time_gps"] = (("time_gps",), time_vals[valid], time_gps_spec.get("CF", {}))

    for v in anchor_present:
        ds[v] = (("time_gps",), flt[v].values[valid], specs.get(v, {}).get("CF", {}))

    companion_vars = [
        v
        for v, s in specs.items()
        if s.get("companion_dim") == "time_gps" and v not in anchor_present
    ]
    for v in companion_vars:
        if v not in flt:
            _log.debug("Companion variable %s not in flight dataset, skipping", v)
            continue
        ds[v] = (("time_gps",), flt[v].values[valid], specs[v].get("CF", {}))
        _log.debug("Added companion variable %s on time_gps dimension", v)

    return ds
