"""Shared pytest fixtures for the glide test suite."""

from importlib import resources

import numpy as np
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


@pytest.fixture(scope="session")
def slocum_l3_2m(tmp_path_factory):
    """L3 dataset binned at 2 m, the grid the microstructure fixtures are built on.

    Four profiles: dive, climb, dive, climb. Yields (dataset, path) so tests can
    use either the in-memory dataset or the file, as the CLI does.
    """
    l2 = str(resources.files("tests").joinpath("data/slocum.l2.nc"))
    out = str(tmp_path_factory.mktemp("l3") / "slocum.2m.l3.nc")

    result = _runner.invoke(app, ["l3", l2, "-o", out, "-b", "2", "-d", "750"])
    assert result.exit_code == 0, f"CLI l3 failed:\n{result.output}"

    ds = xr.open_dataset(out).load()
    yield ds, out
    ds.close()


@pytest.fixture
def make_eps_bin():
    """Factory for a synthetic pyturb eps-bin dataset.

    The real file is 31 MB, so tests build their own. The fixture reproduces the
    parts of the real layout that the merge depends on: 2 m bins centred at
    bin_size/2, per-bin times that run from deep to shallow (an up cast) and sit
    inside chosen L3 profile windows, NaT for bins the profile never reached,
    a deliberately misleading `profile_time` in the preceding profile, variables
    that duplicate ones glide calculates, and a `units: '-'` placeholder.
    """

    def _make(
        l3,
        targets,
        bin_size=2.0,
        n_bins=60,
        fill_from=0,
        coverage=1.0,
        seed=0,
        times=None,
    ):
        rng = np.random.default_rng(seed)
        n = len(targets)
        depth = bin_size / 2 + bin_size * np.arange(n_bins)

        starts = np.asarray(l3.profile_time_start.values).astype("M8[ns]")
        ends = np.asarray(l3.profile_time_end.values).astype("M8[ns]")

        bin_times = np.full((n, n_bins), np.datetime64("NaT", "ns"))
        profile_time = np.empty(n, dtype="M8[ns]")
        for k, target in enumerate(targets):
            span = ends[target] - starts[target]
            n_used = max(2, int(round(n_bins * coverage)))
            for b in range(fill_from, fill_from + n_used):
                if b >= n_bins:
                    break
                # up cast: the shallowest bin is sampled last
                frac = 0.05 + 0.9 * (1 - (b + 0.5) / n_bins)
                bin_times[k, b] = starts[target] + (span * frac).astype("m8[ns]")
            # as in the real files, this points at the previous profile
            profile_time[k] = starts[max(target - 1, 0)]

        if times is not None:
            bin_times = times

        shape = (n, n_bins)
        data = {
            "eps": 10 ** rng.normal(-8.0, 1.0, shape).astype("f4"),
            "eps_1": 10 ** rng.normal(-8.0, 1.0, shape).astype("f4"),
            "chi": 10 ** rng.normal(-9.0, 1.0, shape).astype("f4"),
            "T1": rng.normal(10.0, 1.0, shape).astype("f4"),
            "W": rng.normal(0.35, 0.02, shape).astype("f4"),
            # duplicates quantities glide calculates, so must be refused if declared
            "temperature": rng.normal(10.0, 1.0, shape).astype("f4"),
            "salinity": rng.normal(34.0, 0.1, shape).astype("f4"),
        }
        flags = {
            "eps_qc": rng.choice([0, 1, 2, 4, 9], shape).astype("i1"),
            "chi_qc": rng.choice([1, 2], shape).astype("i1"),
        }

        ds = xr.Dataset(
            data_vars={
                **{k: (("profile", "depth"), v) for k, v in data.items()},
                **{k: (("profile", "depth"), v) for k, v in flags.items()},
                "time": (("profile", "depth"), bin_times),
                "instrument_sn": (("profile",), np.array(["435"] * n)),
            },
            coords={
                "depth": ("depth", depth),
                "profile_time": ("profile", profile_time),
            },
        )
        ds["eps"].attrs = {"long_name": "Best epsilon", "units": "W kg-1"}
        ds["W"].attrs = {"units": "-"}  # the pyturb placeholder
        ds["eps_qc"].attrs = {
            "flag_values": np.array([0, 1, 2, 4, 9], dtype="i1"),
            "flag_meanings": "unknown good questionable bad missing",
        }
        ds.attrs = {
            "instrument_model": "MR1000RDL-EM",
            "pyturb_version": "0.1.0",
            "profile_direction": "up",
        }
        return ds

    return _make
