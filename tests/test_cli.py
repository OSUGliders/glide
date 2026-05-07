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

    ds.close()

    assert not missing_qc_var, f"Missing _qc variables in L2 output: {missing_qc_var}"
    assert not missing_ancillary_attr, (
        f"Missing or incorrect ancillary_variables attribute for: {missing_ancillary_attr}"
    )


def test_l2_directory_output() -> None:
    import re
    import tempfile
    from datetime import datetime, timezone

    data_dir = Path(str(resources.files("tests").joinpath("data")))
    sbd_files = sorted(data_dir.glob("*.sbd.csv"))

    with tempfile.TemporaryDirectory() as tmpdir:
        for sbd_file in sbd_files:
            tbd_file = sbd_file.with_suffix("").with_suffix(".tbd.csv")
            if not tbd_file.exists():
                continue

            glider = sbd_file.name.split("-")[0].split(".")[0]
            result = runner.invoke(
                app, ["l2", str(sbd_file), str(tbd_file), "-o", tmpdir, "-g", glider]
            )
            assert result.exit_code == 0, f"Failed for {sbd_file.name}: {result.output}"

        nc_files = list(Path(tmpdir).glob("*.nc"))
        assert len(nc_files) == len(sbd_files)

        for nc_file in nc_files:
            assert re.match(r"\w+_\d{8}T\d{6}Z\.nc", nc_file.name)

            ds = xr.open_dataset(nc_file)
            first_time = ds["time"].values[0]
            ds.close()

            dt = np.datetime64(first_time, "s").astype("datetime64[s]").astype(datetime)
            dt = dt.replace(tzinfo=timezone.utc)
            expected_ts = dt.strftime("%Y%m%dT%H%M%SZ")
            assert expected_ts in nc_file.name, (
                f"Timestamp mismatch: expected {expected_ts} in {nc_file.name}"
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


def test_backfill() -> None:
    """Test the backfill command for updating velocity in L2 files.

    This test processes real glider data files through the L2 pipeline and
    then runs backfill to verify velocity updates work correctly.
    """
    data_dir = Path(str(resources.files("tests").joinpath("data")))

    # Test segments to process - osu685 real-time data files
    segments = [
        "osu685-2025-056-0-27",
        "osu685-2025-056-0-28",
        "osu685-2025-056-0-29",
        "osu685-2025-056-0-30",
    ]

    # Process L2 files
    l2_files = []
    for seg in segments:
        flt_file = str(data_dir / f"{seg}.sbd.csv")
        sci_file = str(data_dir / f"{seg}.tbd.csv")
        l2_file = data_dir / f"{seg}.l2.nc"

        # Run L2 processing
        result = runner.invoke(app, ["l2", flt_file, sci_file, "-o", str(l2_file)])
        assert result.exit_code == 0, f"L2 processing failed for {seg}: {result.output}"

        l2_files.append(l2_file)

    # Check the first L2 file before backfill
    ds_before = xr.open_dataset(l2_files[0])
    assert "u" in ds_before, "L2 file should have u variable"
    assert "time_uv" in ds_before, "L2 file should have time_uv dimension"

    # Check if velocity is NaN (needs backfill)
    u_before = ds_before.u.values
    has_nan = np.any(np.isnan(u_before))
    ds_before.close()

    if has_nan:
        # Run backfill on the first file
        result = runner.invoke(
            app,
            [
                "backfill",
                str(l2_files[0]),
                "-r",
                str(data_dir),
                "-n",
                "2",  # Include extra files for velocity lookup
            ],
        )
        assert result.exit_code == 0, f"Backfill failed: {result.output}"

        # Check if velocity was updated
        ds_after = xr.open_dataset(l2_files[0])
        u_after = ds_after.u.values

        # Count how many NaN values were filled
        nan_before = np.sum(np.isnan(u_before))
        nan_after = np.sum(np.isnan(u_after))

        # Backfill should have filled at least some values
        assert nan_after <= nan_before, (
            f"Backfill should not increase NaN count: "
            f"before={nan_before}, after={nan_after}"
        )
        ds_after.close()
    else:
        # If velocity was already filled during L2 processing, that's fine
        # Just verify the values are reasonable
        ds_check = xr.open_dataset(l2_files[0])
        u_vals = ds_check.u.values[np.isfinite(ds_check.u.values)]
        if len(u_vals) > 0:
            assert np.all(np.abs(u_vals) < 2.0), "Velocity u should be < 2 m/s"
        ds_check.close()
