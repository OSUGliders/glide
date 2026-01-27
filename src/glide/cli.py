# The command line interface entry point for glide.

import functools
import inspect
import logging
from importlib.metadata import version
from pathlib import Path

import netCDF4 as nc
import numpy as np
import typer
import xarray as xr
from typing_extensions import Annotated

from . import ancillery, config, hotel, process_l1, process_l2, process_l3

_log = logging.getLogger(__name__)

logging.getLogger("flox").setLevel(logging.WARNING)

app = typer.Typer()


def version_callback(value: bool):
    if value:
        typer.echo(f"glide version {version('glide')}")
        raise typer.Exit()


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

    flt_raw = process_l1.parse_l1(flt_file)  # Keep raw for velocity extraction
    flt = process_l1.format_l1(flt_raw.copy(), conf)
    sci = process_l1.parse_l1(sci_file)
    sci = process_l1.format_l1(sci, conf)

    flt = process_l1.apply_qc(flt, conf)
    sci = process_l1.apply_qc(sci, conf)

    merged = process_l1.merge(flt, sci, conf, "science")

    merged = process_l1.calculate_thermodynamics(merged, conf)

    out = process_l1.get_profiles(merged, shallowest_profile, profile_distance)

    out = process_l1.assign_surface_state(out, flt=flt_raw)

    out = process_l1.add_velocity(out, conf, flt=flt_raw)

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

    if dataset_dims == {"time"}:
        input_file_level = 2
    elif dataset_dims == {"profile_id", "z"}:
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
    ds = ancillery.concat(files, concat_dim=concat_dim)

    ds.to_netcdf(out_file)


# @app.command()
# @log_args
# def backfill(
#     flt_file: Annotated[
#         str, typer.Argument(help="Flight file (sbd/dbd) with velocity data.")
#     ],
#     l2_dir: Annotated[
#         str, typer.Argument(help="Directory containing L2 files to update.")
#     ],
#     lookback: Annotated[
#         int, typer.Option("-n", help="Number of previous files to check.")
#     ] = 3,
#     tolerance: Annotated[
#         float, typer.Option("-t", help="Max seconds after dive end.")
#     ] = 300.0,
# ) -> None:
#     """
#     Backfill depth-averaged velocity to previous L2 files missing this data.
#     """
#     l2_path = Path(l2_dir)
#     flt = xr.open_dataset(flt_file, decode_timedelta=True).load()

#     # Extract velocity data
#     if "m_water_vx" not in flt and "m_water_vy" not in flt:
#         typer.echo("No velocity data in flight file.")
#         return

#     time_flt = flt.m_present_time.values
#     u_flt = (
#         flt.m_water_vx.values if "m_water_vx" in flt else np.full_like(time_flt, np.nan)
#     )
#     v_flt = (
#         flt.m_water_vy.values if "m_water_vy" in flt else np.full_like(time_flt, np.nan)
#     )
#     vel_valid = np.isfinite(u_flt) | np.isfinite(v_flt)

#     if not vel_valid.any():
#         typer.echo("No valid velocity data in flight file.")
#         return

#     vel_times = time_flt[vel_valid]
#     vel_u = u_flt[vel_valid]
#     vel_v = v_flt[vel_valid]
#     flt.close()

#     l2_files = sorted(
#         l2_path.glob("*.nc"), key=lambda p: p.stat().st_mtime, reverse=True
#     )
#     l2_files = l2_files[:lookback]

#     updated = []
#     for l2_file in l2_files:
#         with nc.Dataset(str(l2_file), "r+") as ds:
#             if "time_uv" not in ds.variables or "u" not in ds.variables:
#                 continue

#             time_uv = ds.variables["time_uv"][:]
#             u_vals = ds.variables["u"][:]
#             v_vals = ds.variables["v"][:]

#             file_updated = False
#             for i, t_uv in enumerate(time_uv):
#                 if np.isnan(u_vals[i]):
#                     # Find last velocity within tolerance of this dive
#                     match = (vel_times > t_uv - 3600) & (vel_times < t_uv + tolerance)
#                     if match.any():
#                         last_idx = np.where(match)[0][-1]
#                         ds.variables["u"][i] = vel_u[last_idx]
#                         ds.variables["v"][i] = vel_v[last_idx]
#                         file_updated = True
#                         _log.info("Updated dive %d in %s", i, l2_file)

#             if file_updated and str(l2_file) not in updated:
#                 updated.append(str(l2_file))

#     if updated:
#         for f in updated:
#             typer.echo(f"Updated: {f}")
#     else:
#         typer.echo("No files updated.")


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
