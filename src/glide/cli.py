# The command line interface entry point for glide.

import logging

import typer
from typing_extensions import Annotated

from . import level2

_log = logging.getLogger(__name__)

app = typer.Typer()


@app.callback()
def main(log_level: str = "WARN"):
    """Configure the logging level."""
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@app.command()
def l2(
    flt_file: Annotated[str, typer.Argument(help="The flight (dbd/sbd) data file.")],
    sci_file: Annotated[str, typer.Argument(help="The science (ebd/tbd) data file.")],
    out_file: Annotated[str, typer.Argument(help="The output file.")] = "slocum.l2.nc",
    config_file: Annotated[
        str | None,
        typer.Argument(help="Processed variables are specified in this YAML file."),
    ] = None,
):
    _log.debug("Flight file %s", flt_file)
    _log.debug("Science file %s", sci_file)
    _log.debug("Output file %s", out_file)
    _log.debug("Config file %s", config_file)

    config, name_map = level2.load_config(config_file)

    flt = level2.parse_l1(flt_file, config, name_map)
    sci = level2.parse_l1(sci_file, config, name_map)

    out = level2.merge_l1(flt, sci, "science", config)
    out = level2.calculate_thermodynamics(out, config)

    out.to_netcdf(out_file)


@app.command()
def l3():
    return None
