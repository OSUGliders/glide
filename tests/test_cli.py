from importlib import resources

from typer.testing import CliRunner

from glide.cli import app

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


def test_l3() -> None:
    l2_file = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out_file = str(resources.files("tests").joinpath("data/slocum.l3.nc"))
    result = runner.invoke(app, ["l3", l2_file, "-o", out_file])

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
