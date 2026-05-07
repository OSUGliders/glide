# Level 2 processing parses level 1 data produced by dbd2netcdf.
# Some quality control is performed. CF attributes are applied.

import logging
from datetime import datetime, timezone
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from profinder import find_profiles

from . import convert as conv
from . import qc

_log = logging.getLogger(__name__)

# Helper functions


def _fix_time_varaiable_conflict(ds: xr.Dataset) -> xr.Dataset:
    """This fixes conflicting time variable names when parsing a combined flight/science data.
    Generally, they should be parsed separately."""
    if "m_present_time" in ds.variables and "sci_m_present_time" in ds.variables:
        _log.debug(
            "Found conflicting time variables, dropping %s", "sci_m_present_time"
        )
        return ds.drop("sci_m_present_time")
    else:
        return ds


def _format_variables(
    ds: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    """Extracts only variables specified in the config. Applies metadata to variables.
    Converts variable units."""

    _log.debug("Formatting variables")

    if not config.get("slocum"):
        raise ValueError(
            "Configuration has no slocum variable mapping. "
            "Check that core.yml is properly loaded."
        )

    reduced_name_map = {
        var: name for var, name in config["slocum"].items() if var in ds.variables
    }
    for var, name in reduced_name_map.items():
        _log.debug("Formatting variable %s", var)
        specs = config["variables"][name]

        if "conversion" in specs:
            _log.debug("Converting %s with %s", var, specs["conversion"])
            conversion_function = getattr(conv, specs["conversion"])
            ds[var] = (ds[var].dims, conversion_function(ds[var].values), ds[var].attrs)

        if "CF" in specs:
            _log.debug(
                "Applying CF attributes to %s with existing attributes %s",
                var,
                ds[var].attrs,
            )
            ds[var].attrs = specs["CF"]  # Wipes out existing attributes

        _log.debug("Renaming %s to %s", var, name)
        ds = ds.rename({var: name})

    # Drop variables that are not in the specs file.
    remaining_vars = set(ds.keys()) - set(config["variables"].keys())
    ds = ds.drop_vars(remaining_vars)

    if len(ds.data_vars) == 0:
        _log.error(
            "No data variables remain after formatting. Dropped variables: %s. "
            "Expected slocum variables: %s",
            list(remaining_vars),
            list(config["slocum"].keys()),
        )
        raise ValueError("No recognized Slocum variables found in input file. ")

    _log.debug("Variables remaining in dataset %s", list(ds.keys()))

    ds["trajectory"] = (
        "traj_strlen",
        [config["globals"]["trajectory"]["name"]],
        config["globals"]["trajectory"]["attributes"],
    )

    return ds


# Public API functions


def parse_l1(file: str | xr.Dataset) -> xr.Dataset:
    if isinstance(file, str):
        _log.debug("Parsing L1 %s", file)
        try:
            ds = xr.open_dataset(file, decode_timedelta=True).drop_dims("j").load()
            _log.debug("xarray.open_dataset opened %s", file)
        except ValueError:
            ds = pd.read_csv(file).to_xarray()
            # Rename index dimension to 'i' for consistency with NC files
            # Drop the 'i' variable if it exists
            if "i" in ds.data_vars:
                ds = ds.drop_vars("i")
            if "index" in ds.dims:
                ds = ds.rename({"index": "i"})
            _log.debug("pandas.read_csv opened %s", file)
    elif isinstance(file, xr.Dataset):  # Primarily for testing
        ds = file
    else:
        raise ValueError(f"Expected type str or xarray.Dataset but got {type(file)}")
    return ds


def format_l1(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Parses flight (sbd) or science (tbd) data processed by dbd2netcdf or dbd2csv."""

    ds = _fix_time_varaiable_conflict(ds)

    ds = _format_variables(ds, config)

    return ds


def apply_qc(
    ds: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    """The standard suite of L2 QC."""

    ds = qc.init_qc(ds, config=config)

    ds = qc.apply_bounds(ds)

    ds = qc.time(ds)

    # Prior to this point time is a variable and the dimeension is usually `i`.
    dim = list(ds.sizes.keys())[0]
    _log.debug("Swapping dimension %s for time", dim)
    ds = ds.swap_dims({dim: "time"})

    # Applying gps QC will only work on flight data
    # so we need this to catch parsing of science data.
    try:
        ds = qc.gps(ds)
    except AttributeError:
        _log.debug("Failed to apply gps QC.")

    ds = qc.interpolate_missing(ds, config)

    # Drop data that are all nan must come after time is promoted to a coords
    # because we want it to ignore the time coordinate. The time qc dealt with
    # NaNs in the time values.
    dim = list(ds.sizes.keys())[0]
    _log.debug("Before dropna, %i points along dim %s", ds.sizes[dim], dim)
    ds = ds.dropna(dim, how="all")
    _log.debug("After dropna, %i points along dim %s", ds.sizes[dim], dim)

    return ds


def merge(
    flt: xr.Dataset,
    sci: xr.Dataset,
    config: dict,
    times_from: str = "science",
) -> xr.Dataset:
    """Merge flight and science variables onto a common time vector.
    The science time vector is used by default."""

    if times_from == "science":
        time_interpolant = sci.time
        ds_to_interp = flt
        ds = sci
    elif times_from == "flight":
        time_interpolant = flt.time
        ds_to_interp = sci
        ds = flt

    _log.debug("Dims of interpolant are %s", ds.sizes)
    _log.debug("Dims of dataset to interpolate are %s", ds_to_interp.sizes)
    _log.debug("Interpolating onto time from %s", times_from)

    # This dimension gets in the way of interpolation
    if "traj_strlen" in ds_to_interp.dims and "traj_strlen" in ds.dims:
        ds_to_interp = ds_to_interp.drop_dims("traj_strlen")

    vars_to_interp = set(ds_to_interp.variables) - set(ds_to_interp.coords)

    interpolated_vars = []
    for v in vars_to_interp:
        if "_qc" in str(v):
            _log.debug("Skipping %s; QC flags are re-initialized after merge", v)
            continue

        try:  # Only drop variables if the flag is explicitly set
            drop = config["variables"][v]["drop_from_l2"]
            if drop:
                _log.debug("Not interpolating %s due to drop_from_l2 flag in specs", v)
                continue
        except KeyError:
            pass

        _log.debug("Interpolating %s", v)
        ds[v] = (
            "time",
            ds_to_interp[v].interp(time=time_interpolant, assume_sorted=True).values,
            ds_to_interp[v].attrs,
        )
        interpolated_vars.append(str(v))

    # Re-initialize QC for variables interpolated from ds_to_interp that have
    # track_qc: True in the config.  Values are flagged as interpolated (8).
    for v in interpolated_vars:
        if v + "_qc" in ds:
            continue  # Already has a QC variable from the base dataset
        if v not in config["variables"]:
            continue
        if not config["variables"][v].get("track_qc", False):
            continue
        flag_values = np.where(np.isfinite(ds[v].values), np.int8(8), np.int8(9))
        ds = qc.init_qc(ds, v, flag_values)
        _log.debug("Initialized QC for merged variable %s", v)

    _log.debug("Dims interpolated data  %s", ds.sizes)
    _log.debug("Coords interpolated data  %s", list(ds.coords.keys()))
    _log.debug("Variables interpolated data  %s", list(ds.variables.keys()))

    return ds


def calculate_thermodynamics(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Should be applied after merging flight and science.

    Skips calculation if thermo suite is disabled in config.
    """
    if not config.get("include", {}).get("thermo", True):
        _log.debug("Skipping thermodynamics (disabled in config)")
        return ds

    _log.debug("Calculating thermodynamics")

    dims = ds.conductivity.dims

    variable_specs = config["variables"]

    # These variables derive their initial qc from the conductivity_qc.
    salinity = gsw.SP_from_C(
        conv.spm_to_mspcm(ds.conductivity), ds.temperature, ds.pressure
    )
    ds["salinity"] = (dims, salinity.values, variable_specs["salinity"]["CF"])

    lon = ds.lon.interpolate_na("time")
    lat = ds.lat.interpolate_na("time")
    SA = gsw.SA_from_SP(ds.salinity, ds.pressure, lon, lat)
    ds["SA"] = (dims, SA.values, variable_specs["SA"]["CF"])

    density = gsw.rho_t_exact(ds.SA, ds.temperature, ds.pressure)
    ds["density"] = (dims, density.values, variable_specs["density"]["CF"])

    rho0 = gsw.pot_rho_t_exact(ds.SA, ds.temperature, ds.pressure, 0)
    ds["rho0"] = (dims, rho0.values, variable_specs["rho0"]["CF"])

    CT = gsw.CT_from_t(ds.SA, ds.temperature, ds.pressure)
    ds["CT"] = (dims, CT.values, variable_specs["CT"]["CF"])

    sound_speed = gsw.sound_speed(ds.SA, ds.CT, ds.pressure)
    ds["sound_speed"] = (dims, sound_speed.values, variable_specs["sound_speed"]["CF"])

    new_variables = ["salinity", "SA", "density", "rho0", "CT", "sound_speed"]
    ds = qc.init_qc(ds, new_variables, ds.conductivity_qc.values, config)
    ds = qc.apply_bounds(ds, new_variables)

    # These variables derive their initial qc from the pressure_qc so have to be
    # treated separately.
    z = gsw.z_from_p(ds.pressure, lat)
    ds["z"] = (dims, z.values, variable_specs["z"]["CF"])
    ds["depth"] = (dims, -z.values, variable_specs["depth"]["CF"])

    new_variables = ["z", "depth"]
    ds = qc.init_qc(ds, new_variables, ds.pressure_qc.values, config)

    N2, _ = gsw.Nsquared(ds.SA, ds.CT, ds.pressure, ds.lat)

    # N2 is calculated at the mid-point pressures. Here we try interpolating
    # N2 back onto positions of data. This does have the effect of low-pass filtering slightly,
    # which may not be a bad thing because N2 is often noisy.
    N2 = xr.DataArray(N2, {"time": conv.mid(ds.time)})
    ds["N2"] = (dims, N2.interp(time=ds.time).values, variable_specs["N2"]["CF"])
    ds = qc.init_qc(ds, "N2")

    return ds


def get_profiles(
    ds: xr.Dataset, shallowest_profile: float, profile_distance: int
) -> xr.Dataset:
    peaks_kwargs = {
        "height": shallowest_profile,
        "distance": profile_distance,
        "width": profile_distance,
        "prominence": shallowest_profile,
    }

    _log.debug("Finding profiles with peaks_kwargs %s", peaks_kwargs)

    profiles = find_profiles(
        ds.pressure.values,
        peaks_kwargs=peaks_kwargs,
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
) -> xr.Dataset:
    """Assign surface state (0) to unknown points near GPS fixes.

    Points with state == -1 (unknown) that are within `dt` seconds of a valid
    GPS fix are marked as surface state (0).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with state variable from get_profiles.
    flt : xr.Dataset, optional
        Raw flight data containing m_gps_lat/lon with valid GPS fix times.
    dt : float
        Time threshold in seconds for matching to GPS fixes (default 300).

    Returns
    -------
    xr.Dataset
        Dataset with updated state variable.
    """
    if "state" not in ds:
        _log.warning("No state variable in dataset")
        return ds

    if flt is None or "m_gps_lat" not in flt:
        _log.warning("No flight data with GPS, cannot assign surface state")
        return ds

    # Get GPS fix times (where lat is valid)
    gps_valid = np.isfinite(flt.m_gps_lat.values)
    if not gps_valid.any():
        _log.warning("No valid GPS fixes found")
        return ds

    gps_times = flt.m_present_time.values[gps_valid]

    # Update state: unknown (-1) near GPS fixes becomes surface (0)
    state = ds.state.values.copy()
    time_l2 = ds.time.values

    unknown_mask = state == -1
    unknown_indices = np.where(unknown_mask)[0]

    surface_count = 0
    for idx in unknown_indices:
        t = time_l2[idx]
        # Check if any GPS fix is within dt seconds
        if np.any(np.abs(gps_times - t) <= dt):
            state[idx] = 0
            surface_count += 1

    _log.debug("Assigned %d points to surface state", surface_count)

    # Update the state variable
    ds["state"] = (
        "time",
        state,
        ds.state.attrs,
    )

    return ds


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

    # Build groups from velocity reports, each marking a surfacing.
    # Profiles between consecutive velocity reports form one group.
    groups = []  # list of (profile_mask, u_val, v_val)

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

        # Trailing profiles after the last velocity report (no velocity yet)
        mask = (time_l2 > vel_times[-1]) & is_profile
        if mask.any():
            groups.append((mask, np.nan, np.nan))
    else:
        # No velocity data at all — one group with all profiles, NaN velocity
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
        np.isfinite(u).sum(),
    )
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

    # Sort by time (combined L1 files may not be time-ordered)
    order = np.argsort(times)
    times = times[order]
    u_vals = u_vals[order]
    v_vals = v_vals[order]

    # Cluster velocity reports by surfacing events. At each surfacing the
    # flight computer may refine velocity as GPS improves, producing multiple
    # reports within minutes. A gap longer than `gap_threshold` indicates the
    # glider dove and surfaced again. We take the last report from each
    # surfacing as the best estimate.
    gap_threshold = 600  # seconds (10 minutes)
    dt = np.diff(times)
    new_surfacing = np.concatenate(([True], dt > gap_threshold))
    starts = np.where(new_surfacing)[0]
    ends = np.concatenate((starts[1:], [len(times)])) - 1

    _log.debug(
        "Clustered %d velocity reports into %d surfacing events",
        len(times),
        len(ends),
    )

    return times[ends], u_vals[ends], v_vals[ends]


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

    # Anchor variables are always written (they define the coordinate)
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


# Variables that don't belong in IOOS NGDAC profile files
_NGDAC_DROP_VARS = ("dive_id", "climb_id", "state")
_NGDAC_DROP_DIMS = ("time_gps",)
_NGDAC_SCALAR_FROM_TIME_UV = ("u", "v", "time_uv", "lat_uv", "lon_uv")
_NGDAC_INT_FILL = np.int32(-2147483647)

_NGDAC_CRS_ATTRS = {
    "epsg_code": "EPSG:4326",
    "grid_mapping_name": "latitude_longitude",
    "inverse_flattening": 298.257223563,
    "long_name": "http://www.opengis.net/def/crs/EPSG/0/4326",
    "semi_major_axis": 6378137.0,
}

_NGDAC_PROFILE_TIME_ATTRS = {
    "axis": "T",
    "calendar": "gregorian",
    "comment": "Timestamp corresponding to the mid-point of the profile",
    "long_name": "Profile Center Time",
    "observation_type": "calculated",
    "platform": "platform",
    "standard_name": "time",
    "units": "seconds since 1970-01-01T00:00:00Z",
}

_NGDAC_PROFILE_LAT_ATTRS = {
    "axis": "Y",
    "comment": (
        "Value is interpolated to provide an estimate of the latitude "
        "at the mid-point of the profile"
    ),
    "long_name": "Profile Center Latitude",
    "observation_type": "calculated",
    "platform": "platform",
    "standard_name": "latitude",
    "units": "degrees_north",
    "valid_max": 90.0,
    "valid_min": -90.0,
}

_NGDAC_PROFILE_LON_ATTRS = {
    "axis": "X",
    "comment": (
        "Value is interpolated to provide an estimate of the longitude "
        "at the mid-point of the profile"
    ),
    "long_name": "Profile Center Longitude",
    "observation_type": "calculated",
    "platform": "platform",
    "standard_name": "longitude",
    "units": "degrees_east",
    "valid_max": 180.0,
    "valid_min": -180.0,
}


def emit_ioos_profiles(
    ds: xr.Dataset,
    outdir: str | Path,
    glider_name: str,
    instruments: dict | None = None,
    force: bool = False,
) -> list[Path]:
    """Emit one IOOS NGDAC NetCDF file per profile.

    Iterates over each unique non-(-1) ``profile_id`` in the dataset and writes
    one NGDAC-compliant file per profile. Each file:

    * contains only points belonging to that profile (one descent OR one ascent);
    * has scalar ``u``, ``v``, ``time_uv``, ``lat_uv``, ``lon_uv``, and
      ``profile_id`` (no ``time_uv`` dimension);
    * has scalar ``profile_time``, ``profile_lat``, ``profile_lon`` at the
      profile midpoint;
    * has scalar ``platform`` and ``crs`` variables required by NGDAC v2;
    * has one scalar ``instrument_<name>`` variable per entry in
      ``instruments`` carrying its attributes;
    * does not contain ``time_gps``, ``dive_id``, ``climb_id``, or ``state``.

    A profile is skipped (and not written) if its containing segment has no
    finite ``u`` and ``v`` — this is the trailing-segment case where the
    closing surfacing has not yet occurred. The profile will be emitted on a
    future invocation once velocity is reported.

    Profiles within the same surface-to-surface segment share the same
    ``time_uv``, ``lat_uv``, ``lon_uv``, ``u``, and ``v`` values (the canonical
    NGDAC way to identify segment membership).

    Files that already exist in ``outdir`` are skipped silently unless
    ``force=True``.

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset containing ``profile_id``, ``state``, and the standard
        ``time_uv``-dimensioned velocity variables.
    outdir : str or Path
        Directory to write IOOS profile files into. Created if missing.
    glider_name : str
        Glider name, used as the prefix in the IOOS filename
        ``{glider}_{YYYYMMDDTHHMMSSZ}.nc``, and as the ``id`` attribute on
        the ``platform`` variable.
    instruments : dict, optional
        Mapping of instrument variable name (e.g. ``"instrument_ctd"``) to a
        dict of NetCDF attributes (e.g. ``make_model``, ``serial_number``).
        Each becomes a scalar int variable in every emitted file. The
        ``platform`` variable's ``instrument`` attribute is auto-populated
        from these names.
    force : bool, default False
        If True, overwrite existing files instead of skipping them.

    Returns
    -------
    list[Path]
        Paths of files actually written (excluding skipped/preexisting).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for required in ("profile_id", "state"):
        if required not in ds:
            _log.warning("No %s in dataset, cannot emit IOOS files", required)
            return []
    if "time_uv" not in ds.dims:
        _log.warning("No time_uv dimension in dataset, cannot emit IOOS files")
        return []

    profile_ids = np.unique(ds.profile_id.values)
    profile_ids = profile_ids[profile_ids >= 0]

    time_uv_vals = ds.time_uv.values
    u_vals = ds.u.values
    v_vals = ds.v.values
    time_vals = ds.time.values
    state_vals = ds.state.values
    profile_id_vals = ds.profile_id.values
    n = len(state_vals)

    written: list[Path] = []
    for pid in profile_ids:
        pid = int(pid)
        prof_indices = np.where(profile_id_vals == pid)[0]
        if len(prof_indices) == 0:
            continue

        # Walk outward from the profile in state to find the containing segment
        # boundaries. A segment is bounded on each side by either a surface
        # point (state == 0) or the edge of the dataset.
        first_idx = int(prof_indices[0])
        last_idx = int(prof_indices[-1])
        left = first_idx
        while left > 0 and state_vals[left - 1] != 0:
            left -= 1
        right = last_idx
        while right < n - 1 and state_vals[right + 1] != 0:
            right += 1

        seg_time_min = time_vals[left]
        seg_time_max = time_vals[right]
        uv_match = (time_uv_vals >= seg_time_min) & (time_uv_vals <= seg_time_max)
        if not uv_match.any():
            _log.debug("Skipping profile_id %d: no matching time_uv entry", pid)
            continue
        uv_idx = int(np.where(uv_match)[0][0])

        u_val = u_vals[uv_idx]
        v_val = v_vals[uv_idx]
        if not (np.isfinite(u_val) and np.isfinite(v_val)):
            _log.debug(
                "Skipping profile_id %d: u/v not finite (segment awaiting "
                "closing surfacing)",
                pid,
            )
            continue

        prof_mask = profile_id_vals == pid
        prof_times = time_vals[prof_mask]
        first_time = float(prof_times[0])
        timestamp = datetime.fromtimestamp(first_time, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        out_path = outdir / f"{glider_name}_{timestamp}.nc"

        if out_path.exists() and not force:
            _log.debug("Skipping profile_id %d: %s already exists", pid, out_path.name)
            continue

        prof_ds = _slice_profile(ds, prof_mask, uv_idx, pid)
        prof_ds = _add_ngdac_structural_vars(prof_ds, glider_name, instruments)
        prof_ds.to_netcdf(out_path)
        written.append(out_path)
        _log.info(
            "Wrote %s (profile_id=%d, %d points)",
            out_path.name,
            pid,
            int(prof_mask.sum()),
        )

    _log.info("Emitted %d IOOS profile files to %s", len(written), outdir)
    return written


def _slice_profile(
    ds: xr.Dataset,
    prof_mask: np.ndarray,
    uv_idx: int,
    profile_id: int,
) -> xr.Dataset:
    """Build a per-profile NGDAC-shaped dataset.

    Selects the profile's time points, reduces ``time_uv``-dimensioned
    variables to scalars at ``uv_idx``, sets scalar ``profile_id``, and drops
    variables that don't belong in NGDAC profile files (dive_id, climb_id,
    state, time_gps and its companions).
    """
    # Capture scalar values from the source dataset before slicing/dropping.
    scalar_values = {}
    for v in _NGDAC_SCALAR_FROM_TIME_UV:
        if v in ds:
            scalar_values[v] = (ds[v].values[uv_idx], dict(ds[v].attrs))

    pid_attrs = dict(ds["profile_id"].attrs) if "profile_id" in ds else {}

    prof_indices = np.where(prof_mask)[0]
    prof_ds = ds.isel(time=prof_indices)

    # Drop the time_uv dimension entirely (and the variables on it). Scalars
    # will be re-added below with the captured values.
    if "time_uv" in prof_ds.dims:
        prof_ds = prof_ds.drop_dims("time_uv")

    for v, (val, attrs) in scalar_values.items():
        prof_ds[v] = ((), val, attrs)

    if "profile_id" in prof_ds:
        prof_ds = prof_ds.drop_vars("profile_id")
    prof_ds["profile_id"] = ((), np.int32(profile_id), pid_attrs)

    # Drop dims and variables that don't belong in NGDAC profile files.
    for dim in _NGDAC_DROP_DIMS:
        if dim in prof_ds.dims:
            prof_ds = prof_ds.drop_dims(dim)

    for v in _NGDAC_DROP_VARS:
        if v in prof_ds:
            prof_ds = prof_ds.drop_vars(v)

    return prof_ds


def _add_ngdac_structural_vars(
    prof_ds: xr.Dataset,
    glider_name: str,
    instruments: dict | None,
) -> xr.Dataset:
    """Add NGDAC v2 structural scalar variables to a per-profile dataset.

    Adds platform, crs, profile_time, profile_lat, profile_lon, and one
    instrument_<name> scalar per entry in ``instruments``.
    """
    instruments = instruments or {}

    # Profile center: midpoint time, with lat/lon evaluated as the mean of
    # the (already interpolated) per-time arrays. NaN-safe.
    time_vals = prof_ds.time.values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        # Convert to seconds since epoch for arithmetic, then back.
        epoch_sec = time_vals.astype("datetime64[ns]").astype("int64") / 1e9
        profile_time = float(np.nanmean(epoch_sec))
    else:
        profile_time = float(np.nanmean(time_vals.astype("f8")))

    if "lat" in prof_ds:
        profile_lat = float(np.nanmean(prof_ds.lat.values.astype("f8")))
    else:
        profile_lat = float("nan")
    if "lon" in prof_ds:
        profile_lon = float(np.nanmean(prof_ds.lon.values.astype("f8")))
    else:
        profile_lon = float("nan")

    prof_ds["profile_time"] = ((), profile_time, dict(_NGDAC_PROFILE_TIME_ATTRS))
    prof_ds["profile_lat"] = ((), profile_lat, dict(_NGDAC_PROFILE_LAT_ATTRS))
    prof_ds["profile_lon"] = ((), profile_lon, dict(_NGDAC_PROFILE_LON_ATTRS))

    # platform variable. NGDAC requires this; its `instrument` attribute lists
    # the instrument variables present in the file.
    instrument_list = ", ".join(instruments.keys()) if instruments else " "
    platform_attrs = {
        "_FillValue": _NGDAC_INT_FILL,
        "ancillary_variables": " ",
        "comment": "Autonomous vehicle",
        "id": glider_name,
        "instrument": instrument_list,
        "long_name": "platform",
        "type": "platform",
    }
    prof_ds["platform"] = ((), np.int32(0), platform_attrs)

    # CRS variable (boilerplate WGS84).
    crs_attrs = {"_FillValue": _NGDAC_INT_FILL, **_NGDAC_CRS_ATTRS}
    prof_ds["crs"] = ((), np.int32(0), crs_attrs)

    # One scalar per configured instrument.
    for name, attrs in instruments.items():
        full_attrs = {"_FillValue": _NGDAC_INT_FILL, **attrs}
        prof_ds[name] = ((), np.int32(0), full_attrs)

    return prof_ds


def enforce_types(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Enforce data types on variables based on the configuration file."""
    variable_specs = {
        var: specs
        for var, specs in config["variables"].items()
        if var in ds.variables and "dtype" in specs
    }

    for var, specs in variable_specs.items():
        ds[var] = ds[var].astype(specs["dtype"])

    return ds
