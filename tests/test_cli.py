from importlib import resources

from typer.testing import CliRunner

from glide.cli import app

runner = CliRunner()


def test_l2() -> None:
    flt_file = str(resources.files("tests").joinpath("data/osu684.sbd.nc"))
    sci_file = str(resources.files("tests").joinpath("data/osu684.tbd.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    result = runner.invoke(app, ["l2", flt_file, sci_file, "--out-file", out_file])

    assert result.exit_code == 0


def test_l3() -> None:
    l2_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.l3.nc"))
    result = runner.invoke(app, ["l3", l2_file, "--out-file", out_file])

    assert result.exit_code == 0
