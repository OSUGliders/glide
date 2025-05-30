# The command line interface entry point for glide.

import functools
import inspect
import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from . import config, hotel, process_l1, process_l2

_log = logging.getLogger(__name__)

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
def main(log_level: str = "WARN"):
    """Configure the logging level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@app.command()
@log_args
def l1b(
    file: Annotated[str, typer.Argument(help="The sbd/tbd/dbd/ebd data file.")],
    out_file: _out_file_annotation = "slocum.l1b.nc",
    config_file: _config_annotation = None,
) -> None:
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
    output_extras: Annotated[
        bool,
        typer.Option(
            "--extras",
            "-e",
            help="Choose whether to output L1B files (flt/sci/merged).",
        ),
    ] = False,
    p_near_surface: Annotated[
        float, typer.Option(help="Near surface pressure used for detemining profiles.")
    ] = 1.0,
    dp_threshold: Annotated[
        float,
        typer.Option(
            help="Factor applied to determine surfaces when encountering missing data."
        ),
    ] = 5.0,
) -> None:
    conf = config.load_config(config_file)

    flt = process_l1.parse_l1(flt_file, conf)
    sci = process_l1.parse_l1(sci_file, conf)

    flt = process_l1.apply_qc(flt, conf)
    sci = process_l1.apply_qc(sci, conf)

    if output_extras:
        out_dir = Path(out_file).parent
        _log.debug("Saving parsed flight and science output to %s", out_dir)
        flt.to_netcdf(Path(out_dir, "flt.nc"))
        sci.to_netcdf(Path(out_dir, "sci.nc"))

    merged = process_l1.merge(flt, sci, conf, "science")

    merged = process_l1.calculate_thermodynamics(merged, conf)

    if output_extras:
        out_dir = Path(out_file).parent
        _log.debug("Saving merged output to %s", out_dir)
        merged.to_netcdf(Path(out_dir, "merged.nc"))

    out = process_l1.get_profiles(merged, p_near_surface, dp_threshold)

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
    q_netcdf: Annotated[
        str | None,
        typer.Option("--q-in", "-q", help="netCDF file(s) processed by q2netcdf."),
    ] = None,
    config_file: _config_annotation = None,
) -> None:
    l2 = process_l2.parse_l2(l2_file)

    out = process_l2.bin_l2(l2, bin_size)

    if q_netcdf is not None:
        conf = config.load_config(config_file)
        out = process_l2.bin_q(out, q_netcdf, bin_size, conf)

    out.to_netcdf(out_file)


@app.command()
@log_args
def hot(
    l2_file: Annotated[str, typer.Argument(help="The L2 dataset.")],
    out_file: _out_file_annotation = "slocum.hotel.mat",
) -> None:
    l2 = process_l2.parse_l2(l2_file)

    hotel_struct = hotel.create_structure(l2)

    hotel.save_hotel(hotel_struct, out_file)


@app.command()
@log_args
def gps(
    l2_file: Annotated[str, typer.Argument(help="The L2 dataset.")],
    out_file: _out_file_annotation = "slocum.gps.csv",
) -> None:
    l2 = process_l2.parse_l2(l2_file)

    gps = hotel.extract_gps(l2)

    gps.to_dataframe().to_csv(out_file)
