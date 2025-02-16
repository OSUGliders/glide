# Level 2 processing parses level 1 data produced by dbd2netcdf.
# Some quality control is performed. CF attributes are applied.

import logging

import pandas as pd
import xarray as xr
from yaml import safe_load

from . import convert as conv
from . import qc

_log = logging.getLogger(__name__)


def load_variable_specs(file: str | None = None) -> tuple[dict, dict]:
    """Extract variable specifications from a yaml file."""
    if file is None:
        from importlib import resources

        file = str(
            resources.files("glide").joinpath("assets/variable_specification.yml")
        )

    with open(file) as f:
        var_specs = safe_load(f)

    # Generate mapping from Slocum source name to dataset name
    name_map = dict()
    for name, specs in var_specs.items():
        if "source" not in specs:
            continue

        source = specs["source"]
        if type(source) is str:
            name_map[source] = name
            continue

        try:
            for sn in source:
                name_map[sn] = name
        except TypeError:
            continue

    _log.debug("Name mapping dict % s", name_map)

    return var_specs, name_map


def format_variables(
    ds: xr.Dataset,
    var_specs: dict | None = None,
    name_map: dict | None = None,
    var_specs_file: str | None = None,
) -> xr.Dataset:
    """Formats time series variables following the instructions in the variable specification file.
    Drops variables that are not in the file."""

    if var_specs is None or name_map is None:
        var_specs, name_map = load_variable_specs(var_specs_file)

    _log.debug("Formatting variables")
    reduced_name_map = {
        var: name for var, name in name_map.items() if var in ds.variables
    }
    for var, name in reduced_name_map.items():
        _log.debug("Formatting variable %s", var)

        specs = var_specs[name]

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
    remaining_vars = set(ds.keys()) - set(var_specs.keys())
    ds = ds.drop_vars(remaining_vars)

    _log.debug("Variables remaining in dataset %s", list(ds.keys()))

    return ds


def parse_l1(file: str | xr.Dataset, var_specs_file: str | None = None) -> xr.Dataset:
    """Parses flight (sbd) or science (tbd) data processed by dbd2netcdf or dbd2csv.
    Applies IOOS glider DAC attributes."""

    if type(file) is str:
        _log.debug("Parsing L1 %s", file)
        try:
            ds = xr.open_dataset(file).drop_dims("j")
            ds.close()  # TODO: check if needed... I am worried about locking the file
            _log.debug("xarray.open_dataset opened %s", file)
        except ValueError:
            ds = pd.read_csv(file).to_xarray()
            _log.debug("pandas.read_csv opened %s", file)

    elif type(file) is xr.Dataset:  # Primarily for testing
        ds = file
    else:
        raise ValueError(f"Expected type str or xarray.Dataset but got {type(file)}")

    var_specs, name_map = load_variable_specs(var_specs_file)

    # Apply CF conventions, rename variables
    ds = format_variables(ds, var_specs, name_map)

    # Initialize QC flags
    ds = qc.init_dataset_qc(ds, var_specs)

    # Apply time QC
    ds = qc.time(ds)

    # Make time the coordinate.
    dim = list(ds.sizes.keys())[0]  # Assume 1D dataset
    _log.debug("Swapping dimension %s for time", dim)
    ds = ds.swap_dims({dim: "time"})

    # Apply thresholds to all variables using valid min and max from attributes
    ds = qc.apply_data_bounds(ds)

    # Apply gps QC, will only work on flight data
    try:
        ds = qc.gps(ds)
    except AttributeError:
        _log.debug("Failed to apply gps QC.")

    # Interpolate missing data
    ds = qc.interpolate_missing(ds, var_specs)

    # Drop data that are all nan.
    # Must come after promote_time because we want it to ignore the time coordinate.
    dim = list(ds.sizes.keys())[0]
    _log.debug("Before dropna, %i points along dim %s", ds.sizes[dim], dim)
    ds = ds.dropna(dim, how="all")
    _log.debug("After dropna, %i points along dim %s", ds.sizes[dim], dim)

    return ds


def merge_l1(
    flt: xr.Dataset,
    sci: xr.Dataset,
    times_from: str = "science",
    var_specs_file: str | None = None,
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

    var_specs, _ = load_variable_specs(var_specs_file)

    vars_to_interp = set(ds_to_interp.variables) - set(ds_to_interp.coords)

    for v in vars_to_interp:
        if "qc_" in str(v):
            _log.warning("Ignoring %s, merging of QC variables in not currently supported", v)
            continue

        try:  # Only drop variables if the flag is explicitly set
            drop = var_specs[v]["drop_from_l2"]
            if drop:
                _log.debug("Not interpolating %s due to drop_from_l2 flag in specs", v)
                continue
        except KeyError:
            pass

        _log.debug("Iterpolating %s", v)
        ds[v] = (
            "time",
            ds_to_interp[v].interp(time=time_interpolant, assume_sorted=True).values,
            ds_to_interp[v].attrs,
        )

    _log.debug("Dims interpolated data  %s", ds.sizes)
    _log.debug("Coords interpolated data  %s", list(ds.coords.keys()))
    _log.debug("Variables interpolated data  %s", list(ds.variables.keys()))

    return ds
