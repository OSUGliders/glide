from importlib import resources
from pathlib import Path

import numpy as np
import xarray as xr
from typer.testing import CliRunner

from glide.cli import app
from glide.config import load_config

runner = CliRunner()


def test_l1b() -> None:
    flt_file = str(resources.files("tests").joinpath("data/osu684.sbd.csv"))
    out_file = str(resources.files("tests").joinpath("data/slocum.l1b.nc"))
    result = runner.invoke(app, ["l1b", flt_file, "-o", out_file])

    assert result.exit_code == 0


def test_l2() -> None:
    flt_file = str(resources.files("tests").joinpath("data/osu684.sbd.csv"))
    sci_file = str(resources.files("tests").joinpath("data/osu684.tbd.csv"))
    out_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    result = runner.invoke(app, ["l2", flt_file, sci_file, "-o", out_file])

    assert result.exit_code == 0

    # Contract: every variable with track_qc: True in the config that is present
    # in the output must have a _qc counterpart and an ancillary_variables
    # attribute pointing to it.
    config = load_config()
    track_qc_vars = {
        v for v, specs in config["variables"].items() if specs.get("track_qc", False)
    }

    ds = xr.open_dataset(out_file)

    missing_qc_var = []
    missing_ancillary_attr = []
    for var in sorted(track_qc_vars):
        if var not in ds:
            continue
        qc_var = var + "_qc"
        if qc_var not in ds:
            missing_qc_var.append(var)
            continue
        if ds[var].attrs.get("ancillary_variables") != qc_var:
            missing_ancillary_attr.append(var)

    # Contract: surface GPS fixes must appear on a dedicated time_gps dimension
    # with no NaN values (only valid fixes are written).
    assert "time_gps" in ds.dims, "time_gps dimension missing from L2 output"
    assert "lat_gps" in ds, "lat_gps missing from L2 output"
    assert "lon_gps" in ds, "lon_gps missing from L2 output"
    assert ds.sizes["time_gps"] > 0, "Expected at least one GPS fix in L2 output"
    assert np.all(np.isfinite(ds.lat_gps.values)), "lat_gps must not contain NaN"
    assert np.all(np.isfinite(ds.lon_gps.values)), "lon_gps must not contain NaN"

    # Contract: profile_id and segment_id are populated on the time dimension
    # with at least one non-(-1) value (i.e. profiles and segments were found).
    assert "profile_id" in ds, "profile_id missing from L2 output"
    assert "segment_id" in ds, "segment_id missing from L2 output"
    assert (ds.profile_id.values >= 0).any(), "Expected at least one profile"
    assert (ds.segment_id.values >= 0).any(), "Expected at least one segment"
    # All points belonging to a profile must also belong to a segment.
    in_profile = ds.profile_id.values >= 0
    assert (ds.segment_id.values[in_profile] >= 0).all(), (
        "Every profile point must have a non-(-1) segment_id"
    )

    ds.close()

    assert not missing_qc_var, f"Missing _qc variables in L2 output: {missing_qc_var}"
    assert not missing_ancillary_attr, (
        f"Missing or incorrect ancillary_variables attribute for: {missing_ancillary_attr}"
    )


def test_l2_ioos() -> None:
    """Verify --ioos emits NGDAC-shaped per-profile files alongside the merged L2.

    The CLI workflow under test:
        glide l2 <sbd> <tbd> -o <merged.nc> --ioos <outdir> -g <glider>

    Asserts:
      * the merged L2 file is still produced (---ioos is additive)
      * at least one per-profile NGDAC file is emitted to outdir
      * each emitted file has scalar u, v, time_uv, lat_uv, lon_uv,
        profile_id, segment_id (NGDAC v2 contract)
      * each emitted file omits dive_id, climb_id, state, time_gps, time_uv dim
      * filenames follow the NGDAC convention {glider}_{YYYYMMDDTHHMMSSZ}.nc
      * each emitted file's (profile_id, segment_id) pair matches what's in
        the merged L2 file at the same time range
      * re-running the CLI with the same args adds no new files (idempotency)
    """
    import re
    import tempfile

    flt_file = str(resources.files("tests").joinpath("data/osu684.sbd.csv"))
    sci_file = str(resources.files("tests").joinpath("data/osu684.tbd.csv"))

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_file = str(Path(tmpdir) / "slocum.l2.nc")
        ioos_dir = str(Path(tmpdir) / "ioos")

        result = runner.invoke(
            app,
            [
                "l2",
                flt_file,
                sci_file,
                "-o",
                merged_file,
                "--ioos",
                ioos_dir,
                "-g",
                "osu684",
            ],
        )
        assert result.exit_code == 0, result.output

        # --ioos is additive: the merged L2 file is still written.
        assert Path(merged_file).exists(), "Merged L2 file should still be produced"
        merged = xr.open_dataset(merged_file)
        try:
            assert "profile_id" in merged
            assert "segment_id" in merged
        finally:
            merged.close()

        nc_files = sorted(Path(ioos_dir).glob("*.nc"))
        assert len(nc_files) > 0, "Expected at least one IOOS profile file"

        # Cross-check: every (profile_id, segment_id) pair in the IOOS files
        # must also appear in the merged file's per-time arrays.
        merged = xr.open_dataset(merged_file)
        merged_pid = merged.profile_id.values
        merged_seg = merged.segment_id.values
        merged.close()
        merged_pairs = {
            (int(p), int(s))
            for p, s in zip(merged_pid, merged_seg)
            if p >= 0 and s >= 0
        }

        # NGDAC contract checks on every emitted file.
        for nc_file in nc_files:
            assert re.match(r"osu684_\d{8}T\d{6}Z\.nc", nc_file.name), (
                f"Filename does not match NGDAC convention: {nc_file.name}"
            )
            ds = xr.open_dataset(nc_file)
            try:
                # NGDAC requires u, v, time_uv, lat_uv, lon_uv to be scalar.
                for v in ("u", "v", "time_uv", "lat_uv", "lon_uv"):
                    assert v in ds, f"{v} missing from {nc_file.name}"
                    assert ds[v].ndim == 0, (
                        f"{v} must be scalar in {nc_file.name}, got dims {ds[v].dims}"
                    )
                # profile_id and segment_id are scalar identifiers.
                for v in ("profile_id", "segment_id"):
                    assert v in ds, f"{v} missing from {nc_file.name}"
                    assert ds[v].ndim == 0, (
                        f"{v} must be scalar in {nc_file.name}, got dims {ds[v].dims}"
                    )

                # Velocity must be finite (this is the emission gate).
                assert np.isfinite(ds.u.values), f"u is NaN in {nc_file.name}"
                assert np.isfinite(ds.v.values), f"v is NaN in {nc_file.name}"

                # Variables and dims that don't belong in NGDAC profile files.
                for v in ("dive_id", "climb_id", "state", "lat_gps", "lon_gps"):
                    assert v not in ds, f"{v} should not appear in {nc_file.name}"
                assert "time_uv" not in ds.dims, (
                    f"time_uv dim should be removed from {nc_file.name}"
                )
                assert "time_gps" not in ds.dims, (
                    f"time_gps dim should be removed from {nc_file.name}"
                )

                # Cross-check: the (profile_id, segment_id) pair must appear
                # in the merged file's per-time arrays.
                pair = (int(ds.profile_id.values), int(ds.segment_id.values))
                assert pair in merged_pairs, (
                    f"(profile_id={pair[0]}, segment_id={pair[1]}) from "
                    f"{nc_file.name} not found in merged L2 file"
                )
            finally:
                ds.close()

        # Idempotency: re-running must not add new files.
        n_initial = len(nc_files)
        result2 = runner.invoke(
            app,
            [
                "l2",
                flt_file,
                sci_file,
                "-o",
                merged_file,
                "--ioos",
                ioos_dir,
                "-g",
                "osu684",
            ],
        )
        assert result2.exit_code == 0, result2.output
        nc_files_after = sorted(Path(ioos_dir).glob("*.nc"))
        assert len(nc_files_after) == n_initial, (
            f"Re-run added files: {n_initial} -> {len(nc_files_after)}"
        )

        # --force overwrites: file count stays the same but mtimes change.
        first_mtime = nc_files_after[0].stat().st_mtime
        result3 = runner.invoke(
            app,
            [
                "l2",
                flt_file,
                sci_file,
                "-o",
                merged_file,
                "--ioos",
                ioos_dir,
                "-g",
                "osu684",
                "--force",
            ],
        )
        assert result3.exit_code == 0, result3.output
        nc_files_forced = sorted(Path(ioos_dir).glob("*.nc"))
        assert len(nc_files_forced) == n_initial
        assert nc_files_forced[0].stat().st_mtime > first_mtime, (
            "--force should rewrite existing files"
        )


def test_l3() -> None:
    l2_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.l3.nc"))
    result = runner.invoke(
        app, ["l3", l2_file, "-o", out_file, "-b", "10", "-d", "750"]
    )

    assert result.exit_code == 0


def test_hot() -> None:
    l2_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.hotel.mat"))
    result = runner.invoke(app, ["hot", l2_file, "-o", out_file])

    assert result.exit_code == 0


def test_gps() -> None:
    l2_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.gps.csv"))
    result = runner.invoke(app, ["hot", l2_file, "-o", out_file])

    assert result.exit_code == 0


def test_gps_fixes() -> None:
    import pandas as pd

    l2_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.gps_fixes.csv"))
    result = runner.invoke(app, ["gps", l2_file, "-o", out_file, "--fixes"])

    assert result.exit_code == 0, result.output

    df = pd.read_csv(out_file, index_col=0)
    assert "lat_gps" in df.columns
    assert "lon_gps" in df.columns
    assert len(df) > 0
    assert df["lat_gps"].notna().all()
    assert df["lon_gps"].notna().all()
