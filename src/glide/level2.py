# Level 2 processing parses level 1 data produced by dbd2netcdf.
# Some quality control is performed. CF attributes are applied.

import logging

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from . import convert as conv
from . import profiles as pfls
from . import qc

_log = logging.getLogger(__name__)


def parse_config_arg(config: dict | str | None = None) -> dict:
    if isinstance(config, dict):
        pass
    elif isinstance(config, str):
        from .config import load_config

        config = load_config(config)
    elif config is None:
        from .config import load_config

        config = load_config()
    else:
        raise ValueError(
            "config must be configuration dictionary, or string defining the config file path, or None."
        )

    return config


def format_variables(
    ds: xr.Dataset,
    config: dict | str | None = None,
) -> xr.Dataset:
    """Formats time series variables following the instructions in the variable specification file.
    Drops variables that are not in the file."""

    config = parse_config_arg(config)

    _log.debug("Formatting variables")
    reduced_name_map = {
        var: name for var, name in config["slocum"].items() if var in ds.variables
    }
    for var, name in reduced_name_map.items():
        _log.debug("Formatting variable %s", var)
        specs = config["variables"][name]

        if "conversion" in specs:
            _log.debug("Converting %s using %s", var, specs["conversion"])
            conversion_function = getattr(conv, specs["conversion"])
            ds[var] = (ds[var].dims, conversion_function(ds[var].values), ds[var].attrs)

        if "CF" in specs:
            _log.debug(
                "Applying CF attributes to %s which has existing attributes %s",
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


def fix_time_varaiable_conflict(ds: xr.Dataset) -> xr.Dataset:
    if "m_present_time" in ds.variables and "sci_m_present_time" in ds.variables:
        return ds.drop("sci_m_present_time")
    else:
        return ds


def parse_l1(file: str | xr.Dataset) -> xr.Dataset:
    """Parses flight (sbd) or science (tbd) data processed by dbd2netcdf or dbd2csv."""

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


def apply_qc(
    ds: xr.Dataset,
    config: dict | str | None = None,
) -> xr.Dataset:
    """The standard suite of L2 QC."""

    config = parse_config_arg(config)

    ds = fix_time_varaiable_conflict(ds)

    ds = format_variables(ds, config)

    ds = qc.init_qc(ds, config)

    ds = qc.apply_bounds(ds)

    ds = qc.time(ds)

    # Make time the coordinate.
    dim = list(ds.sizes.keys())[0]  # Assume 1D dataset
    _log.debug("Swapping dimension %s for time", dim)
    ds = ds.swap_dims({dim: "time"})

    # Apply gps QC, will only work on flight data
    try:
        ds = qc.gps(ds)
    except AttributeError:
        _log.debug("Failed to apply gps QC.")

    # Interpolate missing data
    ds = qc.interpolate_missing(ds, config)

    # Drop data that are all nan.
    # Must come after promote_time because we want it to ignore the time coordinate.
    dim = list(ds.sizes.keys())[0]
    _log.debug("Before dropna, %i points along dim %s", ds.sizes[dim], dim)
    ds = ds.dropna(dim, how="all")
    _log.debug("After dropna, %i points along dim %s", ds.sizes[dim], dim)

    return ds


def merge(
    flt: xr.Dataset,
    sci: xr.Dataset,
    times_from: str = "science",
    config: dict | str | None = None,
) -> xr.Dataset:
    """Merge flight and science variables onto a common time vector.
    The science time vector is used by default."""

    config = parse_config_arg(config)

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
            drop = config[v]["drop_from_l2"]
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

    dims = ds.conductivity.dims

    variable_specs = config["variables"]

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

    # Initialize quality control
    new_variables = ["salinity", "SA", "density", "rho0", "CT"]
    ds = qc.init_qc(ds, config, new_variables, ds.conductivity_qc.values)
    ds = qc.apply_bounds(ds, new_variables)

    z = gsw.z_from_p(ds.pressure, lat)
    ds["z"] = (dims, z.values, variable_specs["z"]["CF"])
    ds["depth"] = (dims, -z.values, variable_specs["depth"]["CF"])

    new_variables = ["z", "depth"]
    ds = qc.init_qc(ds, config, new_variables, ds.pressure_qc.values)

    N2, _ = gsw.Nsquared(ds.SA, ds.CT, ds.pressure, ds.lat)
    # Try interpolating N2 back onto positions of data.
    # This does have the effect of low-pass filtering.
    N2 = xr.DataArray(N2, {"time": conv.mid(ds.time)})
    ds["N2"] = (dims, N2.interp(time=ds.time).values, variable_specs["N2"]["CF"])
    ds = qc.init_qc_variable(ds, "N2")

    return ds


def get_profiles(
    ds: xr.Dataset, p_near_surface: float, dp_threshold: float
) -> xr.Dataset:
    dive_id, climb_id, state = pfls.find_profiles_using_logic(
        ds.pressure, p_near_surface, dp_threshold
    )
    ds["dive_id"] = ("time", dive_id, dict(_FillValue=-1))
    ds["climb_id"] = ("time", climb_id, dict(_FillValue=-1))
    ds["state"] = (
        "time",
        state.astype("b"),
        dict(
            long_name="Glider state",
            flag_values=np.array([-1, 0, 1, 2], "b"),
            flag_meanings="state_unknown surfaced diving climbing",
            valid_max=np.int8(3),
            valid_min=np.int8(-1),
        ),
    )
    return ds
