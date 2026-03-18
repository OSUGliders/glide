# Level 2 processing parses level 1 data produced by dbd2netcdf.
# Some quality control is performed. CF attributes are applied.

import logging

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

    for v in vars_to_interp:
        if "_qc" in str(v):
            _log.warning(
                "Ignoring %s, merging of QC variables in not currently supported", v
            )
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

    new_variables = ["salinity", "SA", "density", "rho0", "CT"]
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
    state = np.full(n, -1, dtype="b")

    dive_counter = 1
    climb_counter = 1

    for prof in profiles:
        dive_start, dive_end, climb_start, climb_end = prof

        dive_id[dive_start:dive_end] = dive_counter
        state[dive_start:dive_end] = 1
        dive_counter += 1

        climb_id[climb_start:climb_end] = climb_counter
        state[climb_start:climb_end] = 2
        climb_counter += 1

    ds["dive_id"] = ("time", dive_id, dict(_FillValue=np.int32(-1)))
    ds["climb_id"] = ("time", climb_id, dict(_FillValue=np.int32(-1)))
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


def _find_dive_cycles(state: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find dive cycles from state array.

    A dive cycle is a contiguous sequence of dive (1), climb (2), and unknown (-1)
    states bounded by surface (0) states. Unknown states are included because
    there may be gaps at inflection points between dive and climb.

    Parameters
    ----------
    state : np.ndarray
        State array with values -1 (unknown), 0 (surface), 1 (dive), 2 (climb).

    Returns
    -------
    list of tuples
        Each tuple is (cycle_start, cycle_end, surf_start, surf_end) where:
        - cycle_start/end: indices of the dive/climb portion
        - surf_start/end: indices of the following surface period (for velocity lookup)
    """
    cycles = []
    n = len(state)
    i = 0

    while i < n:
        # Skip initial unknown states at the very start
        while i < n and state[i] == -1:
            i += 1

        if i >= n:
            break

        # Skip surface period (before dive)
        while i < n and state[i] == 0:
            i += 1

        if i >= n:
            break

        # Now we should be in dive/climb or unknown (gap)
        # Look for any dive or climb state to start a cycle
        if state[i] not in (-1, 1, 2):
            i += 1
            continue

        # Find the actual start of dive/climb activity
        cycle_start = i

        # Continue through dive (1), climb (2), and unknown (-1) until we hit surface (0)
        # This captures the full dive cycle including any gaps at inflection points
        while i < n and state[i] != 0:
            i += 1

        cycle_end = i

        # Check if this segment actually contains any dive or climb
        segment_states = state[cycle_start:cycle_end]
        has_dive_climb = np.any((segment_states == 1) | (segment_states == 2))

        if not has_dive_climb:
            # Pure unknown segment, skip it
            continue

        # Find following surface period
        surf_start = i
        while i < n and state[i] == 0:
            i += 1
        surf_end = i

        # Only record if we found a surface period after (for velocity)
        if surf_end > surf_start:
            cycles.append((cycle_start, cycle_end, surf_start, surf_end))
        else:
            # No surface after - this is incomplete, skip velocity
            cycles.append((cycle_start, cycle_end, -1, -1))

    return cycles


def add_velocity(
    ds: xr.Dataset,
    config: dict,
    flt: xr.Dataset | None = None,
) -> xr.Dataset:
    """Add depth-averaged velocity variables to L2 dataset.

    Uses state transitions to identify dive cycles. A dive cycle is all
    contiguous dive (1) and climb (2) states between surface (0) periods.
    Velocity is taken from the following surface period.

    Parameters
    ----------
    ds : xr.Dataset
        L2 dataset with state variable (must have surface states assigned).
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

    if "state" not in ds:
        _log.warning("No state variable in dataset - run get_profiles first")
        return _empty_velocity()

    # Find dive cycles from state transitions
    cycles = _find_dive_cycles(ds.state.values)
    n_cycles = len(cycles)

    if n_cycles < 1:
        _log.warning("No dive cycles detected from state transitions")
        return _empty_velocity()

    _log.debug("Found %d dive cycles from state transitions", n_cycles)

    # Extract velocity data from flight data
    vel_times, vel_u, vel_v = _extract_velocity_data(flt)

    time_l2 = ds.time.values
    lat_l2 = ds.lat.values
    lon_l2 = ds.lon.values

    time_uv = np.full(n_cycles, np.nan)
    lat_uv = np.full(n_cycles, np.nan)
    lon_uv = np.full(n_cycles, np.nan)
    u = np.full(n_cycles, np.nan)
    v = np.full(n_cycles, np.nan)

    for i, (cycle_start, cycle_end, surf_start, surf_end) in enumerate(cycles):
        # Mean time and position from the dive/climb portion
        cycle_times = time_l2[cycle_start:cycle_end]
        cycle_lats = lat_l2[cycle_start:cycle_end]
        cycle_lons = lon_l2[cycle_start:cycle_end]

        if len(cycle_times) > 0:
            time_uv[i] = np.nanmean(cycle_times)
            lat_uv[i] = np.nanmean(cycle_lats)
            lon_uv[i] = np.nanmean(cycle_lons)

        # Find velocity in the following surface period
        if surf_start < 0 or vel_times is None:
            continue

        t_start = time_l2[surf_start]
        t_end = time_l2[surf_end - 1] + 60 if surf_end > surf_start else t_start + 60

        u_vel, v_vel = _find_velocity_in_window(vel_times, vel_u, vel_v, t_start, t_end)
        if u_vel is not None:
            u[i] = u_vel
            v[i] = v_vel
            _log.debug("Cycle %d: assigned u=%.4f, v=%.4f", i, u[i], v[i])

    ds["time_uv"] = (("time_uv",), time_uv, specs["time_uv"]["CF"])
    ds["lat_uv"] = (("time_uv",), lat_uv, specs["lat_uv"]["CF"])
    ds["lon_uv"] = (("time_uv",), lon_uv, specs["lon_uv"]["CF"])
    ds["u"] = (("time_uv",), u, specs["u"]["CF"])
    ds["v"] = (("time_uv",), v, specs["v"]["CF"])

    _log.info(
        "Added velocity for %d dive cycles, %d with valid data",
        n_cycles,
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

    return time_flt[vel_valid], u_flt[vel_valid], v_flt[vel_valid]


def _find_velocity_in_window(
    vel_times: np.ndarray | None,
    vel_u: np.ndarray | None,
    vel_v: np.ndarray | None,
    t_start: float,
    t_end: float,
) -> tuple[float | None, float | None]:
    """Find the last valid velocity estimate in a time window.

    Parameters
    ----------
    vel_times : array
        Times of valid velocity estimates.
    vel_u, vel_v : array
        Eastward and northward velocity values.
    t_start, t_end : float
        Time window to search (inclusive start, exclusive end).

    Returns
    -------
    tuple of (u, v) or (None, None) if no velocity found.
    """
    if vel_times is None or vel_u is None or vel_v is None:
        return None, None

    match = (vel_times >= t_start) & (vel_times < t_end)
    if match.any():
        last_idx = np.where(match)[0][-1]
        return vel_u[last_idx], vel_v[last_idx]

    return None, None


def backfill_velocity(
    l2_file: str,
    raw_files: list[str],
    tolerance: float = 0.005,
) -> bool:
    """Backfill velocity data in an L2 file using raw flight data.

    For each velocity estimate, searches raw flight data for velocity in a
    time window starting at the end of the climb (last point with matching
    climb_id after time_uv). Updates the value if it is missing (NaN) or if
    the new estimate differs significantly from the existing one.

    Parameters
    ----------
    l2_file : str
        Path to L2 file to update.
    raw_files : list of str
        Paths to raw flight files (sbd/dbd) containing velocity data.
    tolerance : float
        Update existing velocity if difference exceeds this value (m/s).

    Returns
    -------
    bool
        True if the file was updated, False otherwise.
    """
    import netCDF4 as nc

    raw_datasets = []
    for rf in raw_files:
        raw_datasets.append(parse_l1(rf))

    if not raw_datasets:
        _log.warning("No raw data could be loaded for backfill")
        return False

    flt = xr.concat(raw_datasets, dim="i")
    vel_times, vel_u, vel_v = _extract_velocity_data(flt)

    if vel_times is None:
        return False

    _log.debug("Found %d velocity estimates in raw data", len(vel_times))

    # Update the L2 file
    file_updated = False
    with nc.Dataset(l2_file, "r+") as ds:
        if "u" not in ds.variables or "v" not in ds.variables:
            _log.warning("No velocity variables in %s", l2_file)
            return False

        if "climb_id" not in ds.variables:
            _log.warning("No climb_id variable in %s", l2_file)
            return False

        time_l2 = ds.variables["time"][:]
        time_uv = np.ma.filled(ds.variables["time_uv"][:], np.nan)
        climb_id = np.ma.filled(ds.variables["climb_id"][:], -1)
        u_vals = np.ma.filled(ds.variables["u"][:], np.nan)
        v_vals = np.ma.filled(ds.variables["v"][:], np.nan)

        n_uv = len(time_uv)
        if n_uv == 0:
            _log.debug("No time_uv values in %s", l2_file)
            return False

        for i in range(n_uv):
            t_uv = time_uv[i]
            if np.isnan(t_uv):
                continue  # No valid time_uv for this cycle

            # Find the last point with this climb_id after time_uv
            climb_mask = (climb_id == i) & (time_l2 >= t_uv)
            if not climb_mask.any():
                _log.debug("No climb points after time_uv for cycle %d", i)
                continue

            # Search window starts at end of climb
            t_start = time_l2[climb_mask][-1]

            # Search window ends at next valid time_uv (or +10 min)
            if i + 1 < n_uv and np.isfinite(time_uv[i + 1]):
                t_end = time_uv[i + 1]
            else:
                t_end = t_start + 600  # 10 minutes max

            u_vel, v_vel = _find_velocity_in_window(
                vel_times, vel_u, vel_v, t_start, t_end
            )
            if u_vel is None:
                continue

            # Check if update is needed (missing or significantly different)
            if u_vel is None or v_vel is None:
                continue

            u_old, v_old = u_vals[i], v_vals[i]
            is_missing = np.isnan(u_old) or np.isnan(v_old)
            is_different = not is_missing and (
                abs(u_vel - u_old) > tolerance or abs(v_vel - v_old) > tolerance
            )

            if is_missing or is_different:
                ds.variables["u"][i] = u_vel
                ds.variables["v"][i] = v_vel
                file_updated = True
                if is_missing:
                    _log.info("Backfilled cycle %d: u=%.4f, v=%.4f", i, u_vel, v_vel)
                else:
                    _log.info(
                        "Updated cycle %d: u=%.4f->%.4f, v=%.4f->%.4f",
                        i,
                        u_old,
                        u_vel,
                        v_old,
                        v_vel,
                    )

    return file_updated


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
