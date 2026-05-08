import logging

import numpy as np
import xarray as xr
from profinder import find_profiles

_log = logging.getLogger(__name__)


def _detect_drift(
    pressure: np.ndarray,
    time: np.ndarray,
    dt_s: float,
    flt: xr.Dataset | None = None,
    min_drift_duration: float = 600.0,
    pressure_std_threshold: float = 2.0,
    surface_pressure: float = 2.0,
) -> np.ndarray:
    """Return boolean mask True where glider is drifting at depth.

    Uses x_hover_active from raw flight data if available (reported as 1 when
    a hover episode starts and 0 when it ends, with -127 as the missing-data
    sentinel; the most recent reported value is forward-filled to every science
    timestamp).  Falls back to a rolling-variance detector on the pressure
    signal: a candidate region must have rolling pressure std <
    pressure_std_threshold and pressure > surface_pressure sustained for at
    least min_drift_duration seconds.
    """
    n = len(pressure)
    drift = np.zeros(n, dtype=bool)

    if flt is not None and "x_hover_active" in flt:
        flt_time = flt.m_present_time.values
        hover_active = flt.x_hover_active.values
        # -127 is the no-data sentinel for the signed-byte variable; treat as
        # missing along with NaN. Forward-fill the most recent {0,1} report.
        finite = (
            np.isfinite(hover_active) & np.isfinite(flt_time) & (hover_active != -127)
        )
        if finite.any():
            flt_time = flt_time[finite]
            hover_active = hover_active[finite]
            order = np.argsort(flt_time)
            flt_time = flt_time[order]
            hover_active = hover_active[order]
            pos = np.searchsorted(flt_time, time, side="right") - 1
            in_range = pos >= 0
            drift[in_range] = hover_active[pos[in_range]] == 1
            _log.debug(
                "Drift: %d points from x_hover_active (%d reports)",
                int(drift.sum()),
                len(flt_time),
            )
            return drift
        _log.debug("x_hover_active has no finite values; using pressure detector")

    valid = np.isfinite(pressure)
    if not valid.any():
        return drift

    # Window = min_drift_duration / 4; smaller window reduces edge miss at drift boundaries.
    # np.convolve mode='same' returns max(n, window) elements, so cap half so window < n.
    half = min((n - 1) // 2, max(2, round(min_drift_duration / 4 / dt_s)))
    window = 2 * half + 1
    ones = np.ones(window)

    p_fill = np.where(valid, pressure, 0.0)
    cnt_w = np.convolve(valid.astype(float), ones, mode="same")
    sum_w = np.convolve(p_fill, ones, mode="same")
    sumsq_w = np.convolve(p_fill**2, ones, mode="same")

    safe_cnt = np.where(cnt_w >= 3, cnt_w, 1.0)
    mean_w = sum_w / safe_cnt
    var_w = sumsq_w / safe_cnt - mean_w**2
    rolling_std = np.where(cnt_w >= 3, np.sqrt(np.maximum(0.0, var_w)), np.inf)

    candidate = (
        valid & (pressure > surface_pressure) & (rolling_std < pressure_std_threshold)
    )

    # Keep only contiguous runs that span at least min_drift_duration
    i = 0
    while i < n:
        if not candidate[i]:
            i += 1
            continue
        j = i + 1
        while j < n and candidate[j]:
            j += 1
        if (j - i) * dt_s >= min_drift_duration:
            drift[i:j] = True
        i = j

    _log.debug("Drift: %d points from pressure-based detector", int(drift.sum()))
    return drift


def _absorb_apex_unknowns(
    state: np.ndarray,
    dive_id: np.ndarray,
    climb_id: np.ndarray,
    profile_id: np.ndarray,
    pressure: np.ndarray,
    dt_s: float,
    max_gap_duration: float,
    surface_pressure: float = 2.0,
) -> None:
    """Absorb short unknown gaps at the underwater apex of a multi-yo segment.

    profinder occasionally leaves a few samples between dive_end and
    climb_start unclassified at the peak.  When such a gap is short, sits at
    depth (max pressure > surface_pressure), and is sandwiched directly
    between a dive (1) and a climb (2), split it at the pressure maximum:
    points up to and including the max are folded into the preceding dive,
    points after into the following climb.
    """
    n = len(state)
    unknown = state == -1
    if not unknown.any():
        return

    diff = np.diff(unknown.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    max_samples = max(2, round(max_gap_duration / dt_s))

    for s, e in zip(starts, ends):
        if (e - s) > max_samples:
            continue
        if s == 0 or e >= n:
            continue
        if state[s - 1] != 1 or state[e] != 2:
            continue
        p = pressure[s:e]
        finite = np.isfinite(p)
        if not finite.any() or float(np.min(p[finite])) < surface_pressure:
            continue

        local_max = int(np.argmax(np.where(finite, p, -np.inf)))
        split = s + local_max + 1

        state[s:split] = 1
        dive_id[s:split] = dive_id[s - 1]
        profile_id[s:split] = profile_id[s - 1]

        state[split:e] = 2
        climb_id[split:e] = climb_id[e]
        profile_id[split:e] = profile_id[e]


def _absorb_post_drift_transients(
    state: np.ndarray,
    dive_id: np.ndarray,
    climb_id: np.ndarray,
    profile_id: np.ndarray,
    drift_mask: np.ndarray,
    dt_s: float,
    max_transient_duration: float,
) -> None:
    """Fold brief dive/unknown segments between a drift end and the next climb
    into that climb.

    Masking the drift period leaves a discontinuous pressure signal across the
    gap, which occasionally causes profinder to detect a spurious tiny peak
    just after the drift in addition to the real climb.  This pass extends the
    climb backward to the drift end if the only intervening states are short
    dive (1) or unknown (-1) segments.  Anything longer than
    max_transient_duration past the drift end is left alone.
    """
    n = len(state)
    if not drift_mask.any():
        return

    drift_diff = np.diff(drift_mask.astype(int), prepend=0, append=0)
    drift_ends = np.where(drift_diff == -1)[0]
    window = max(2, round(max_transient_duration / dt_s))

    for de in drift_ends:
        if de >= n:
            continue
        end = min(de + window, n)
        in_window = state[de:end]
        climb_local = np.where(in_window == 2)[0]
        if len(climb_local) == 0:
            continue
        first_climb = de + int(climb_local[0])
        if first_climb == de:
            continue
        intermediate = state[de:first_climb]
        if not np.all((intermediate == 1) | (intermediate == -1)):
            continue
        cid = climb_id[first_climb]
        pid = profile_id[first_climb]
        state[de:first_climb] = 2
        climb_id[de:first_climb] = cid
        profile_id[de:first_climb] = pid
        dive_id[de:first_climb] = -1


def get_profiles(
    ds: xr.Dataset,
    shallowest_profile: float,
    min_surface_time: float = 180.0,
    flt: xr.Dataset | None = None,
    min_drift_duration: float = 600.0,
    drift_pressure_std: float = 2.0,
    stall_tolerance: float = 180.0,
    min_pressure_rate: float = 0.05,
) -> xr.Dataset:
    """Identify dive, climb, and drift profiles from a pressure time series.

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
    flt : xr.Dataset, optional
        Raw flight data.  Used to detect drift-at-depth via BAW_HOVER_ACTIVE
        when present; otherwise pressure-based detection is used.
    min_drift_duration : float
        Minimum duration (seconds) for a constant-pressure period to be
        classified as drift (default 600 s).
    drift_pressure_std : float
        Rolling pressure standard deviation threshold (dbar) for pressure-based
        drift detection (default 2.0 dbar).
    stall_tolerance : float
        Maximum time (seconds) the glider can stall or briefly reverse during
        a dive or climb without profinder terminating the profile (default
        180 s = 3 min, sized to handle realistic glider stalls during
        ascent).  Converted to profinder's per-sample `run_length` argument.
    min_pressure_rate : float
        Minimum pressure change rate (dbar/s) for a sample to count toward
        ascent or descent classification (default 0.05 dbar/s ≈ 5 cm/s, well
        below normal glider vertical speed of 10–20 cm/s).  Converted to
        profinder's per-sample `min_pressure_change` argument.
    """
    raw_diff = np.diff(ds.time.values)
    if np.issubdtype(raw_diff.dtype, np.timedelta64):
        dt_s = float(np.nanmedian(raw_diff.astype("timedelta64[s]").astype("f8")))
    else:
        dt_s = float(np.nanmedian(raw_diff))
    fs = 1.0 / dt_s

    drift_mask = _detect_drift(
        ds.pressure.values,
        ds.time.values,
        dt_s,
        flt=flt,
        min_drift_duration=min_drift_duration,
        pressure_std_threshold=drift_pressure_std,
    )

    # Mask drift before profile finding so profinder sees a gap across the
    # drift period; with missing="drop" it will pair the real climb after
    # drift with the correct dive.
    pressure_masked = ds.pressure.values.copy().astype(float)
    pressure_masked[drift_mask] = np.nan

    peaks_kwargs = {
        "height": shallowest_profile,
        "prominence": shallowest_profile,
        "distance": max(4, round(min_surface_time * fs)),
        "width": max(2, round(20 * fs)),  # ≥20 s half-width detects ~10 m dives
    }
    troughs_kwargs = {
        # Real inter-profile troughs (surfacings or inter-yo apexes) have a
        # depth contrast at least as large as a real dive's prominence; a
        # smaller "trough" is a stall or noise dip within a profile.
        "prominence": shallowest_profile,
        "distance": max(2, round(30 * fs)),
        "width": max(1, round(5 * fs)),
    }

    _log.debug(
        "fs=%.3f Hz; peaks_kwargs=%s; troughs_kwargs=%s",
        fs,
        peaks_kwargs,
        troughs_kwargs,
    )

    # Scale stall tolerance and pressure-rate threshold to per-sample units.
    # These guard the climb/dive run from being terminated by brief stalls
    # (e.g. glider hangs at depth and drifts back down a few dbar before
    # resuming the ascent).
    run_length = max(2, round(stall_tolerance / dt_s))
    min_pressure_change = float(min_pressure_rate * dt_s)

    profiles = find_profiles(
        pressure_masked,
        peaks_kwargs=peaks_kwargs,
        troughs_kwargs=troughs_kwargs,
        run_length=run_length,
        min_pressure_change=min_pressure_change,
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

    # Override drift points, clearing any profile membership assigned above
    state[drift_mask] = 3
    dive_id[drift_mask] = -1
    climb_id[drift_mask] = -1
    profile_id[drift_mask] = -1

    _absorb_post_drift_transients(
        state, dive_id, climb_id, profile_id, drift_mask, dt_s, min_surface_time
    )

    _absorb_apex_unknowns(
        state,
        dive_id,
        climb_id,
        profile_id,
        ds.pressure.values,
        dt_s,
        min_surface_time,
    )

    ds["dive_id"] = ("time", dive_id, dict(_FillValue=np.int32(-1)))
    ds["climb_id"] = ("time", climb_id, dict(_FillValue=np.int32(-1)))
    ds["profile_id"] = ("time", profile_id, dict(_FillValue=np.int32(-1)))
    ds["state"] = (
        "time",
        state,
        dict(
            long_name="Glider state",
            flag_values=np.array([-1, 0, 1, 2, 3], "b"),
            flag_meanings="unknown surface dive climb drift",
            valid_max=np.int8(3),
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
    """Assign surface state (0) to unknown points.

    Two checks are applied in sequence:

    1. A point with state == -1 (unknown) is marked as surface if it is within
       `dt` seconds of a valid GPS fix AND shallower than `surface_pressure`
       dbar.  The pressure gate prevents the broad GPS time window from
       reaching into adjacent dives.
    2. Any remaining contiguous unknown segment is marked as surface unless
       its pressure record contains a finite value at or above
       `surface_pressure` (i.e. evidence the glider was deeper).  Missing
       pressure is the expected state at the surface — the science sensor
       is often out of water — so all-NaN unknown segments are classified
       as surface.  This covers the period before the first dive, after the
       last climb, and any between-profile gap.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with state and pressure variables from get_profiles.
    flt : xr.Dataset, optional
        Raw flight data containing m_gps_lat/lon with valid GPS fix times.
        If absent, only the pressure-based check runs.
    dt : float
        Time threshold in seconds for matching to GPS fixes (default 300).
    surface_pressure : float
        Maximum pressure (dbar) for a point to be considered at the surface
        (default 2.0).
    """
    state = ds.state.values.copy()
    pressure = ds.pressure.values
    time_l2 = ds.time.values

    if flt is not None and "m_gps_lat" in flt:
        gps_valid = np.isfinite(flt.m_gps_lat.values)
        if gps_valid.any():
            gps_times = np.sort(flt.m_present_time.values[gps_valid])
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
                is_shallow = np.isfinite(pressure[unknown_mask]) & (
                    pressure[unknown_mask] < surface_pressure
                )
                state[unknown_mask] = np.where(
                    is_near & is_shallow, np.int8(0), state[unknown_mask]
                )
                _log.debug(
                    "GPS-proximity surface check: %d points assigned",
                    int((is_near & is_shallow).sum()),
                )
        else:
            _log.warning("No valid GPS fixes found")
    else:
        _log.warning("No flight data with GPS, skipping GPS-proximity surface check")

    # Whole-segment pressure check: any contiguous unknown run that never
    # exceeds surface_pressure is the glider sitting at the surface
    unknown = state == -1
    if unknown.any():
        n = len(state)
        diff = np.diff(unknown.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        n_assigned = 0
        for s, e in zip(starts, ends):
            p = pressure[s:e]
            finite = np.isfinite(p)
            # Skip if there is finite evidence the glider was underwater.
            if finite.any() and float(np.max(p[finite])) >= surface_pressure:
                continue
            # An all-NaN gap bounded by dive→climb is the underwater apex
            # that the apex absorber couldn't split because pressure was
            # missing.  Leave it unknown rather than mislabel as surface.
            if (
                not finite.any()
                and 0 < s
                and e < n
                and state[s - 1] == 1
                and state[e] == 2
            ):
                continue
            state[s:e] = 0
            n_assigned += int(e - s)
        _log.debug("Whole-segment surface check: %d points assigned", n_assigned)

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
