# The command line interface entry point for glide.

import functools
import inspect
import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from . import config, hotel, process_l1, process_l2, process_l3

_log = logging.getLogger(__name__)

logging.getLogger("flox").setLevel(logging.WARNING)

app = typer.Typer()


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
def main(log_level: str = "WARN", log_file: str | None = None) -> None:
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

    ds = process_l1.parse_l1(file, conf)

    ds = process_l1.apply_qc(ds, conf)

    ds.to_netcdf(out_file)


@app.command()
@log_args
def l2(
    flt_file: Annotated[str, typer.Argument(help="The flight (dbd/sbd) data file.")],
    sci_file: Annotated[str, typer.Argument(help="The science (ebd/tbd) data file.")],
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
) -> None:
    """
    Generate L2 data from L1 data.
    """
    conf = config.load_config(config_file)

    flt = process_l1.parse_l1(flt_file, conf)
    sci = process_l1.parse_l1(sci_file, conf)

    flt = process_l1.apply_qc(flt, conf)
    sci = process_l1.apply_qc(sci, conf)

    merged = process_l1.merge(flt, sci, conf, "science")

    merged = process_l1.calculate_thermodynamics(merged, conf)

    out = process_l1.get_profiles(merged, shallowest_profile, profile_distance)

    out = process_l1.enforce_types(out, conf)

    out.attrs = conf["globals"]["netcdf_attributes"]

    out.encoding["unlimited_dims"] = {}

    out.to_netcdf(out_file)


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
    l2 = process_l2.parse_l2(l2_file)

    out = process_l2.bin_l2(l2, bin_size, depth)

    if q_netcdf is not None:
        conf = config.load_config(config_file)

        q = process_l3.parse_q(q_netcdf)

        out = process_l3.bin_q(out, q, bin_size, conf)

    out.to_netcdf(out_file)


@app.command()
@log_args
def ml3(
    l3_file: Annotated[str, typer.Argument(help="The L3 dataset.")],
    out_file: _out_file_annotation = "slocum.l3.nc",
    q_netcdf: Annotated[
        str | None,
        typer.Option("--q-in", "-q", help="netCDF file(s) processed by q2netcdf."),
    ] = None,
    config_file: _config_annotation = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            "-w",
            help="Overwrite the existing L3 dataset if it exists.",
        ),
    ] = False,
) -> None:
    """
    Merge ancillary data into L3 data.
    """
    # I could remove the defaul argument to enforce this rule but I am anticipating that
    # in the future we may want to merge other kinds of data into the L3 dataset.
    if q_netcdf is None:
        raise typer.BadParameter("The --q-in option is required for ml3 command.")

    if not overwrite and Path(out_file).exists():
        raise typer.BadParameter(
            f"The output file {out_file} already exists. Use --overwrite to overwrite it."
        )

    l3, bin_size = process_l3.parse_l3(l3_file)

    conf = config.load_config(config_file)

    q = process_l3.parse_q(q_netcdf)

    out = process_l3.bin_q(l3, q, bin_size, conf)

    out.to_netcdf(out_file)


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
) -> None:
    """
    Generate gps csv file from L2 data.
    """
    l2 = process_l2.parse_l2(l2_file)

    gps = hotel.extract_gps(l2)

    gps.to_dataframe().to_csv(out_file)


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
    ds = process_l3.concat(files, concat_dim=concat_dim)

    ds.to_netcdf(out_file)


@app.command()
@log_args
def cfg(
    out_file: _out_file_annotation = "default.config.yaml",
) -> None:
    """
    Output the default configuration file.
    """
    import shutil
    from importlib import resources

    config_file = str(resources.files("glide").joinpath("assets/config.yml"))
    shutil.copy(config_file, out_file)
