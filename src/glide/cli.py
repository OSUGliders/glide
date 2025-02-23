# The command line interface entry point for glide.

import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from . import level2, level3

_log = logging.getLogger(__name__)

app = typer.Typer()


@app.callback()
def main(log_level: str = "WARN"):
    """Configure the logging level."""
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@app.command()
def l1b(
    file: Annotated[str, typer.Argument(help="The sbd/tbd/dbd/ebd data file.")],
    out_file: Annotated[
        str, typer.Option("--out", "-o", help="The output file.")
    ] = "slocum.l1b.nc",
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Processed variables are specified in this YAML file.",
        ),
    ] = None,
) -> None:
    _log.debug("Input file %s", file)
    _log.debug("Output file %s", out_file)
    _log.debug("Config file %s", config_file)

    config, name_map = level2.load_config(config_file)

    out = level2.parse_l1(file)

    out = level2.apply_qc(out, config, name_map)

    out.to_netcdf(out_file)


@app.command()
def l2(
    flt_file: Annotated[str, typer.Argument(help="The flight (dbd/sbd) data file.")],
    sci_file: Annotated[str, typer.Argument(help="The science (ebd/tbd) data file.")],
    out_file: Annotated[
        str, typer.Option("--out", "-o", help="The output file.")
    ] = "slocum.l2.nc",
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Processed variables are specified in this YAML file.",
        ),
    ] = None,
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
    _log.debug("Flight file %s", flt_file)
    _log.debug("Science file %s", sci_file)
    _log.debug("Output file %s", out_file)
    _log.debug("Config file %s", config_file)

    config, name_map = level2.load_config(config_file)

    flt = level2.parse_l1(flt_file)
    sci = level2.parse_l1(sci_file)

    flt = level2.apply_qc(flt, config, name_map)
    sci = level2.apply_qc(sci, config, name_map)

    if output_extras:
        out_dir = Path(out_file).parent
        _log.debug("Saving parsed flight and science output to %s", out_dir)
        flt.to_netcdf(Path(out_dir, "flt.nc"))
        sci.to_netcdf(Path(out_dir, "sci.nc"))

    merged = level2.merge(flt, sci, "science", config)

    merged = level2.calculate_thermodynamics(merged, config)

    if output_extras:
        out_dir = Path(out_file).parent
        _log.debug("Saving merged output to %s", out_dir)
        merged.to_netcdf(Path(out_dir, "merged.nc"))

    out = level2.get_profiles(merged, p_near_surface, dp_threshold)

    out.to_netcdf(out_file)


@app.command()
def l3(
    l2_file: Annotated[str, typer.Argument(help="The L2 dataset.")],
    out_file: Annotated[
        str, typer.Option("--out", "-o", help="The output file.")
    ] = "slocum.l3.nc",
    bin_size: Annotated[
        float, typer.Option("--bin", "-b", help="Depth bin size in meters.")
    ] = 10.0,
    q_netcdf: Annotated[
        str | None,
        typer.Option("--q-in", "-q", help="netCDF file(s) processed by q2netcdf."),
    ] = None,
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config",
            "-c",
            help="Processed variables are specified in this YAML file.",
        ),
    ] = None,
) -> None:
    _log.debug("L2 file %s", l2_file)
    _log.debug("Output file %s", out_file)
    _log.debug("Bin size %s", bin_size)
    _log.debug("q netcdf file %s", q_netcdf)

    l2 = level3.parse_l2(l2_file)

    out = level3.bin_l2(l2, bin_size)

    if q_netcdf is not None:
        config, _ = level2.load_config(config_file)
        out = level3.bin_q(out, q_netcdf, bin_size, config)

    out.to_netcdf(out_file)

    return None
