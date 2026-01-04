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

    _log.debug("Variables remaining in dataset %s", list(ds.keys()))
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
    """Should be applied after merging flight and science."""

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

    ds["dive_id"] = ("time", dive_id, dict(_FillValue=np.int16(-1)))
    ds["climb_id"] = ("time", climb_id, dict(_FillValue=np.int16(-1)))
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
