# The command line interface entry point for glide.

import functools
import inspect
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import netCDF4 as nc
import typer
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


def generate_ioos_filename(ds, glider_name: str) -> str:
    first_time = float(ds["time"].values[0])
    dt = datetime.fromtimestamp(first_time, tz=timezone.utc)
    timestamp = dt.strftime("%Y%m%dT%H%M%SZ")
    return f"{glider_name}_{timestamp}.nc"


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
            help="Glider name for IOOS-style filename when output is a directory.",
        ),
    ] = None,
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

    out_path = Path(out_file)
    if out_path.is_dir():
        name = glider_name or conf["globals"]["trajectory"]["name"].split("_")[-1]
        out_path = out_path / generate_ioos_filename(out, name)

    out.to_netcdf(out_path)

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


@app.command()
@log_args
def backfill(
    l2_files: Annotated[
        list[str], typer.Argument(help="L2 files to check for missing velocity.")
    ],
    raw_dir: Annotated[
        str,
        typer.Option("-r", "--raw-dir", help="Directory containing raw sbd/dbd files."),
    ],
    extra_files: Annotated[
        int,
        typer.Option(
            "-n",
            "--extra",
            help="Number of extra raw files to load after last L2 file.",
        ),
    ] = 3,
) -> None:
    """
    Backfill depth-averaged velocity to L2 files.

    Uses the glider state variable in L2 files to identify dive cycles, then
    looks up velocity from the corresponding raw flight files. Updates velocity
    if missing or if the new estimate differs significantly from the existing one.

    File naming convention: L2 files should be named like 'basename.l2.nc'
    where 'basename.sbd.nc' or 'basename.dbd.nc' is the corresponding raw file.
    """
    raw_path = Path(raw_dir)

    # Sort L2 files by name to ensure chronological order
    l2_files_sorted = sorted([Path(f) for f in l2_files])

    # Filter to files that have velocity variables
    files_to_update = []
    for l2_file in l2_files_sorted:
        try:
            with nc.Dataset(str(l2_file), "r") as ds:
                if "time_uv" in ds.variables:
                    files_to_update.append(l2_file)
        except Exception as e:
            _log.warning("Could not read %s: %s", l2_file, e)
            continue

    if not files_to_update:
        typer.echo("No L2 files with time_uv variable found.")
        return

    typer.echo(f"Processing {len(files_to_update)} L2 files.")

    # Extract base names from L2 files (remove .l2.nc suffix)
    def get_base_name(l2_file: Path) -> str:
        name = l2_file.name
        for suffix in [".l2.nc", ".L2.nc", ".nc"]:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name.rsplit(".", 1)[0]

    first_base = get_base_name(files_to_update[0])
    last_base = get_base_name(files_to_update[-1])

    # Get all sbd/dbd files in raw directory, sorted
    # Look for .sbd.nc, .sbd.csv, .dbd.nc, .dbd.csv patterns
    raw_files = sorted(
        list(raw_path.glob("*.sbd.nc"))
        + list(raw_path.glob("*.sbd.csv"))
        + list(raw_path.glob("*.dbd.nc"))
        + list(raw_path.glob("*.dbd.csv"))
    )

    if not raw_files:
        typer.echo(f"No raw flight files found in {raw_dir}")
        return

    # Extract base names from raw files (remove .sbd.nc, .sbd.csv, etc.)
    def get_raw_base_name(raw_file: Path) -> str:
        name = raw_file.name
        for suffix in [
            ".sbd.nc",
            ".sbd.csv",
            ".dbd.nc",
            ".dbd.csv",
            ".tbd.nc",
            ".tbd.csv",
        ]:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name.rsplit(".", 1)[0]

    raw_names = [get_raw_base_name(f) for f in raw_files]

    # Find indices of first and last files to load using exact base name match
    try:
        first_idx = raw_names.index(first_base)
    except ValueError:
        typer.echo(f"Could not find raw file matching {first_base}")
        return

    try:
        last_idx = raw_names.index(last_base)
    except ValueError:
        last_idx = first_idx

    # Load from first to last + extra_files
    end_idx = min(last_idx + extra_files + 1, len(raw_files))
    raw_files_to_load = [str(f) for f in raw_files[first_idx:end_idx]]

    typer.echo(f"Loading {len(raw_files_to_load)} raw flight files...")

    # Update each L2 file
    updated = []
    for l2_file in files_to_update:
        if process_l1.backfill_velocity(str(l2_file), raw_files_to_load):
            updated.append(str(l2_file))

    if updated:
        typer.echo(f"\nUpdated {len(updated)} files:")
        for f in updated:
            typer.echo(f"  {f}")
    else:
        typer.echo("No files were updated.")


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
