"""Shared pytest fixtures for the glide test suite."""

from importlib import resources

import pytest
import xarray as xr
from typer.testing import CliRunner

from glide.cli import app

_runner = CliRunner()


@pytest.fixture(scope="session")
def sl685_l2(tmp_path_factory):
    """L2 xr.Dataset produced by running the CLI over the sl685 test fixtures.

    Built once per test session by invoking ``glide l2`` on the trimmed
    sl685.dbd.csv / sl685.ebd.csv files in tests/data/.  All tests that need
    a realistic L2 dataset (e.g. flight model calibration) should use this
    fixture rather than loading a pre-baked CSV.
    """
    dbd = str(resources.files("tests").joinpath("data/sl685.dbd.csv"))
    ebd = str(resources.files("tests").joinpath("data/sl685.ebd.csv"))
    out = str(tmp_path_factory.mktemp("l2") / "sl685.l2.nc")

    result = _runner.invoke(app, ["l2", dbd, ebd, "-o", out])
    assert result.exit_code == 0, f"CLI l2 failed:\n{result.output}"

    ds = xr.open_dataset(out).load()
    yield ds
    ds.close()
