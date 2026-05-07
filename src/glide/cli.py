# The command line interface entry point for glide.

import functools
import inspect
import logging
from importlib.metadata import version
from pathlib import Path

import netCDF4 as nc
import typer
import xarray as xr
from typing import Annotated

from . import (
    ancillery,
    config,
    gliderdac,
    hotel,
    process_l1,
    process_l2,
    process_l3,
    profiles,
)

_log = logging.getLogger(__name__)

logging.getLogger("flox").setLevel(logging.WARNING)

app = typer.Typer()


def version_callback(value: bool):
    if value:
        typer.echo(f"glide version {version('glide')}")
        raise typer.Exit()


def _concat_raw(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate raw L1 datasets along their index dim.

    parse_l1 returns datasets with an `i` index dim. Single-file calls
    return the dataset unchanged; multi-file calls stack along `i`.
    """
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(
        datasets, dim="i", data_vars="minimal", coords="minimal", compat="override"
    )


def log_args(func):
    """Decorator to log all argument names and values to a function using the logging module."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        _log.debug("Calling %s", func.__name__)
        for name, value in bound.arguments.items():
            _log.debug("  %s = %r", name, value)
        return func(*args, **kwargs)

    return wrapper


# Commonly used argument annotations
_config_annotation = Annotated[
    str | None,
    typer.Option(
        "--config",
        "-c",
        help="Processing configuration is specified in this YAML file.",
    ),
]
_out_file_annotation = Annotated[
    str, typer.Option("--out", "-o", help="The output file.")
]


@app.callback()
def main(
    log_level: str = "WARN",
    log_file: str | None = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v", callback=version_callback, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """glide is a command line program for processing Slocum glider data."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        filename=log_file,
    )


@app.command()
@log_args
def l1b(
    file: Annotated[str, typer.Argument(help="The sbd/tbd/dbd/ebd data file.")],
    out_file: _out_file_annotation = "slocum.l1b.nc",
    config_file: _config_annotation = None,
) -> None:
    """
    Generate L1B data from L1 data.
    """
    conf = config.load_config(config_file)

    ds = process_l1.parse_l1(file)

    ds = process_l1.format_l1(ds, conf)

    ds = process_l1.apply_qc(ds, conf)

    ds.to_netcdf(out_file)


@app.command()
@log_args
def l2(
    flt_file: Annotated[
        str,
        typer.Argument(
            help="The flight (sbd/dbd) data file or a glob pattern matching "
            "multiple flight files. Quote the pattern to prevent shell "
            "expansion (e.g. '*.sbd.csv').",
        ),
    ],
    sci_file: Annotated[
        str,
        typer.Argument(
            help="The science (tbd/ebd) data file or a glob pattern matching "
            "multiple science files. Each science file must pair with a "
            "flight file of the same stem.",
        ),
    ],
    out_file: _out_file_annotation = "slocum.l2.nc",
    config_file: _config_annotation = None,
    shallowest_profile: Annotated[
        float, typer.Option("-s", help="Shallowest allowed profile in dbar.")
    ] = 5.0,
    profile_distance: Annotated[
        int,
        typer.Option(
            "-d", help="Minimum distance between profiles in number of data points."
        ),
    ] = 20,
    skip_unpaired: Annotated[
        bool,
        typer.Option(
            "--skip-unpaired",
            help="If set, drop flight or science files that have no partner "
            "(matched by basename stem) with a warning instead of failing.",
        ),
    ] = False,
    riot_csv: Annotated[
        str | None,
        typer.Option(
            "-r",
            "--riot-csv",
            help="File path to output a RIOT-compatible CSV file in addition "
            "to netCDF.",
        ),
    ] = None,
    riot_add_positions: Annotated[
        bool,
        typer.Option(
            "--riot-positions",
            help="Interpolate and add depth, latitude, and longitude into RIOT CSV "
            "output.",
        ),
    ] = False,
    glider_name: Annotated[
        str | None,
        typer.Option(
            "--glider",
            "-g",
            help="Glider name used as the prefix in IOOS profile filenames. "
            "Defaults to the trailing component of the trajectory name in the "
            "config.",
        ),
    ] = None,
    ioos_dir: Annotated[
        str | None,
        typer.Option(
            "--ioos",
            help="If set, additionally emit one IOOS NGDAC NetCDF file per "
            "profile into this directory. Profiles whose containing segment "
            "has no finite velocity (i.e. trailing dives without a closing "
            "surfacing) are skipped and will be emitted on a future run.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="When emitting IOOS profile files, overwrite existing files "
            "instead of skipping them.",
        ),
    ] = False,
) -> None:
    """
    Generate L2 data from L1 data.
    """
    conf = config.load_config(config_file)

    pairs = process_l1.pair_input_files(flt_file, sci_file, skip_unpaired)
    _log.info("Processing %d flight/science file pair(s)", len(pairs))

    flt_raw = _concat_raw([process_l1.parse_l1(f) for f, _ in pairs])
    sci_raw = _concat_raw([process_l1.parse_l1(s) for _, s in pairs])

    flt = process_l1.format_l1(flt_raw.copy(), conf)
    sci = process_l1.format_l1(sci_raw, conf)

    flt = process_l1.apply_qc(flt, conf)
    sci = process_l1.apply_qc(sci, conf)

    merged = process_l1.merge(flt, sci, conf, "science")

    merged = process_l1.calculate_thermodynamics(merged, conf)

    out = profiles.get_profiles(merged, shallowest_profile, profile_distance)

    out = profiles.assign_surface_state(out, flt=flt_raw)

    out = profiles.add_velocity(out, conf, flt=flt_raw)

    out = profiles.add_gps_fixes(out, flt, conf)

    out = process_l1.enforce_types(out, conf)

    out.attrs = {
        k: v for k, v in conf["globals"]["netcdf_attributes"].items() if v is not None
    }

    out.encoding["unlimited_dims"] = {}

    out.to_netcdf(out_file)

    if ioos_dir is not None:
        name = glider_name or conf["globals"]["trajectory"]["name"].split("_")[-1]
        gliderdac.emit_ioos_profiles(
            out,
            ioos_dir,
            name,
            instruments=conf.get("instruments", {}),
            force=force,
            ngdac=conf.get("ngdac"),
        )

    if riot_csv:
        from .riot_csv_writer import write_riot_csv

        write_riot_csv(out, riot_add_positions, riot_csv)


@app.command()
@log_args
def l3(
    l2_file: Annotated[str, typer.Argument(help="The L2 dataset.")],
    out_file: _out_file_annotation = "slocum.l3.nc",
    bin_size: Annotated[
        float, typer.Option("--bin", "-b", help="Depth bin size in meters.")
    ] = 10.0,
    depth: Annotated[
        float | None, typer.Option("--depth", "-d", help="Max depth for binning.")
    ] = None,
    q_netcdf: Annotated[
        str | None,
        typer.Option("--q-in", "-q", help="netCDF file(s) processed by q2netcdf."),
    ] = None,
    config_file: _config_annotation = None,
) -> None:
    """
    Generate L3 data from L2 data.
    """
    conf = config.load_config(config_file)

    l2 = process_l2.parse_l2(l2_file)

    out = process_l2.bin_l2(l2, bin_size, depth, conf)

    if q_netcdf is not None:
        q = ancillery.parse_q(q_netcdf)

        out = process_l3.bin_q(out, q, bin_size, conf)

    out.to_netcdf(out_file)


@app.command()
@log_args
def merge(
    glide_file: Annotated[
        str, typer.Argument(help="A L2 or L3 dataset produced by glide.")
    ],
    input_file: Annotated[str, typer.Argument(help="Input file(s) of a given type.")],
    file_type: Annotated[
        str,
        typer.Argument(
            help="Choose 'q' for q2netcdf output file, 'p' for p for p2netcdf output file."
        ),
    ],
    out_file: _out_file_annotation = "slocum.merged.nc",
    config_file: _config_annotation = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            "-w",
            help="Overwrite the existing dataset if it exists.",
        ),
    ] = False,
) -> None:
    """
    Merge ancillary data into L2 or L3 data.
    """

    if file_type not in ["q", "p"]:
        raise typer.BadParameter(f"The file type {file_type} must be q or p.")

    if Path(out_file).exists() and not overwrite:
        raise typer.BadParameter(
            f"The output file {out_file} already exists. Use --overwrite to overwrite it."
        )

    # Figure out the processig level of the input
    input_file_level = -1
    ds = nc.Dataset(glide_file)
    dataset_dims = set(ds.dimensions)
    ds.close()

    # L2 files have a time dimension and may have additional per-event dimensions
    # (time_uv for velocity, time_gps for GPS fixes, traj_strlen for trajectory).
    # L3 files are indexed by profile_id and z.
    _L2_DIMS = {"time", "time_uv", "time_gps", "traj_strlen"}
    _L3_DIMS = {"profile_id", "z"}

    if "time" in dataset_dims and dataset_dims <= _L2_DIMS:
        input_file_level = 2
    elif _L3_DIMS <= dataset_dims:
        input_file_level = 3
    else:
        raise ValueError(
            f"Could not determine processing level of input file {glide_file} with dimensions {dataset_dims}"
        )

    conf = config.load_config(config_file)

    if file_type == "q":
        if input_file_level == 3:
            l3, bin_size = process_l3.parse_l3(glide_file)
            q = ancillery.parse_q(input_file)
            out = process_l3.bin_q(l3, q, bin_size, conf)
            out.to_netcdf(out_file)
        else:
            raise NotImplementedError(
                "Merging q files only supported for level 3 data."
            )
    if file_type == "p":
        raise NotImplementedError("Merging of p files is not yet supported.")


@app.command()
@log_args
def hot(
    l2_file: Annotated[str, typer.Argument(help="The L2 dataset.")],
    out_file: _out_file_annotation = "slocum.hotel.mat",
) -> None:
    """
    Generate hotel mat file from L2 data.
    """
    l2 = process_l2.parse_l2(l2_file)

    hotel_struct = hotel.create_structure(l2)

    hotel.save_hotel(hotel_struct, out_file)


@app.command()
@log_args
def gps(
    l2_file: Annotated[str, typer.Argument(help="The L2 dataset.")],
    out_file: _out_file_annotation = "slocum.gps.csv",
    fixes_only: Annotated[
        bool,
        typer.Option(
            "--fixes",
            help="Extract surface GPS fixes on the time_gps dimension instead of "
            "interpolated positions on the science time grid.",
        ),
    ] = False,
) -> None:
    """
    Generate gps csv file from L2 data.
    """
    l2 = process_l2.parse_l2(l2_file)

    if fixes_only:
        out = hotel.extract_gps_fixes(l2)
    else:
        out = hotel.extract_gps(l2)

    out.to_dataframe().to_csv(out_file)


@app.command()
@log_args
def concat(
    files: Annotated[
        list[str], typer.Argument(help="The netcdf files to concatenate.")
    ],
    out_file: _out_file_annotation = "concat.nc",
    concat_dim: Annotated[
        str,
        typer.Option("--concat-dim", "-d", help="The dimension to concatenate along."),
    ] = "time",
) -> None:
    """
    Concatenate multiple netCDF files along a dimension.
    """
    ds = ancillery.concat(files, concat_dim=concat_dim)

    ds.to_netcdf(out_file)


@app.command()
@log_args
def cfg(
    out_file: _out_file_annotation = "default.config.yml",
) -> None:
    """
    Output the default configuration file.
    """
    import shutil
    from importlib import resources

    config_file = str(resources.files("glide").joinpath("assets/config.yml"))
    shutil.copy(config_file, out_file)
