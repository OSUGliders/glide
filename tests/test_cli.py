from importlib import resources

from typer.testing import CliRunner

from glide.cli import app

runner = CliRunner()


def test_app() -> None:
    flt_file = str(resources.files("tests").joinpath("data/osu684.sbd.nc"))
    sci_file = str(resources.files("tests").joinpath("data/osu684.tbd.nc"))
    result = runner.invoke(app, ["l2", flt_file, sci_file])
    assert result.exit_code == 0
