# Level 2 processing parses level 1 data produced by dbd2netcdf.
# Some quality control is performed. CF attributes are applied.

import glob
import logging
import re
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from . import convert as conv
from . import qc

_log = logging.getLogger(__name__)

_FLT_INFIXES = ("sbd", "dbd")
_SCI_INFIXES = ("tbd", "ebd")
_INPUT_EXTS = ("nc", "csv")
_FILE_RE = re.compile(
    r"^(?P<stem>.+)\.(?P<infix>"
    + "|".join(_FLT_INFIXES + _SCI_INFIXES)
    + r")\.(?P<ext>"
    + "|".join(_INPUT_EXTS)
    + r")$"
)


# Helper functions


def _fix_time_variable_conflict(ds: xr.Dataset) -> xr.Dataset:
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


def _classify_file(path: str) -> tuple[str, str]:
    """Return ('flt'|'sci', stem) for a recognized glider data file path.

    Raises ValueError if the basename does not match
    `<stem>.(sbd|dbd|tbd|ebd).(nc|csv)`.
    """
    name = Path(path).name
    match = _FILE_RE.match(name)
    if match is None:
        raise ValueError(
            f"{path!r} does not look like a glider data file "
            f"(expected <stem>.(sbd|dbd|tbd|ebd).(nc|csv))"
        )
    infix = match.group("infix")
    kind = "flt" if infix in _FLT_INFIXES else "sci"
    return kind, match.group("stem")


def pair_input_files(
    flt_pattern: str,
    sci_pattern: str,
    skip_unpaired: bool = False,
) -> list[tuple[str, str]]:
    """Expand glob patterns and pair flight files to science files by stem.

    Each pattern may be a literal filename or a shell-style glob. Every flight
    file in the expansion of `flt_pattern` must classify as flight
    (.sbd/.dbd) and every science file must classify as science (.tbd/.ebd).

    Pairing is by basename stem, e.g. `glider-2025-001.sbd.csv` pairs with
    `glider-2025-001.tbd.csv`. Stems must match exactly.

    By default, raises ValueError if any flight or science file is unpaired.
    With `skip_unpaired=True`, unpaired files are dropped and a warning is
    logged.
    """
    flt_paths = sorted(glob.glob(flt_pattern)) or (
        [flt_pattern] if Path(flt_pattern).exists() else []
    )
    sci_paths = sorted(glob.glob(sci_pattern)) or (
        [sci_pattern] if Path(sci_pattern).exists() else []
    )

    if not flt_paths:
        raise ValueError(f"No flight files matched pattern {flt_pattern!r}")
    if not sci_paths:
        raise ValueError(f"No science files matched pattern {sci_pattern!r}")

    flt_by_stem: dict[str, str] = {}
    for p in flt_paths:
        kind, stem = _classify_file(p)
        if kind != "flt":
            raise ValueError(
                f"{p!r} matched the flight pattern but is not a flight file "
                f"(expected .sbd or .dbd)"
            )
        if stem in flt_by_stem:
            raise ValueError(
                f"Duplicate flight stem {stem!r}: {flt_by_stem[stem]!r} and {p!r}"
            )
        flt_by_stem[stem] = p

    sci_by_stem: dict[str, str] = {}
    for p in sci_paths:
        kind, stem = _classify_file(p)
        if kind != "sci":
            raise ValueError(
                f"{p!r} matched the science pattern but is not a science file "
                f"(expected .tbd or .ebd)"
            )
        if stem in sci_by_stem:
            raise ValueError(
                f"Duplicate science stem {stem!r}: {sci_by_stem[stem]!r} and {p!r}"
            )
        sci_by_stem[stem] = p

    flt_only = sorted(set(flt_by_stem) - set(sci_by_stem))
    sci_only = sorted(set(sci_by_stem) - set(flt_by_stem))
    paired_stems = sorted(set(flt_by_stem) & set(sci_by_stem))

    if flt_only or sci_only:
        msg_parts = []
        if flt_only:
            msg_parts.append(
                f"flight files without a science partner: "
                f"{[flt_by_stem[s] for s in flt_only]}"
            )
        if sci_only:
            msg_parts.append(
                f"science files without a flight partner: "
                f"{[sci_by_stem[s] for s in sci_only]}"
            )
        msg = "Unpaired input files; " + "; ".join(msg_parts)
        if skip_unpaired:
            _log.warning("%s. Dropping them.", msg)
        else:
            raise ValueError(msg + ". Pass --skip-unpaired to drop them.")

    if not paired_stems:
        raise ValueError(
            "No flight/science file pairs found; "
            "every flight file must have a science file with the same stem."
        )

    return [(flt_by_stem[s], sci_by_stem[s]) for s in paired_stems]


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

    ds = _fix_time_variable_conflict(ds)

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
    # The old index dim survives as a non-dimension coord, drop it.
    if dim in ds.coords:
        ds = ds.drop_vars([dim])

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
