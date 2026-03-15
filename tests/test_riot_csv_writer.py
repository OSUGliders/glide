"""Tests for the riot_csv_writer module and the --riot-csv / --riot-positions
CLI options on the ``l2`` command.

This file is self-contained so it can be removed cleanly if the tests
are no longer wanted.
"""
# Author: Claude Opus 4.6 and Stuart Pearce

import os
import textwrap

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from glide.riot_csv_writer import write_riot_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_riot_dataset(n: int = 5, include_positions: bool = False) -> xr.Dataset:
    """Build a minimal xr.Dataset that satisfies write_riot_csv requirements.

    Parameters
    ----------
    n : int
        Number of time steps.
    include_positions : bool
        If True, add depth / lat / lon variables so that ``_add_positions``
        has something to interpolate.
    """
    time = np.arange(n, dtype=np.float64) + 1.0  # non-zero epoch seconds

    ds = xr.Dataset(
        {
            "sr_ping_epoch_days": ("time", np.full(n, 19500, dtype=np.float64)),
            "sr_ping_secs": ("time", np.arange(n, dtype=np.float64) * 10),
            "sr_ping_msecs": ("time", np.arange(n, dtype=np.float64) * 100),
            "sr_ping_rt_msecs": ("time", np.arange(n, dtype=np.float64) * 1000),
            "sr_ping_freq": ("time", np.full(n, 69000, dtype=np.float64)),
            "sr_ping_detection_level": (
                "time",
                np.random.default_rng(0).integers(0, 100, n).astype(np.float64),
            ),
            "sr_ping_sequence_number": ("time", np.arange(n, dtype=np.float64)),
            "sr_ping_platform_id": ("time", np.full(n, 42, dtype=np.float64)),
            "sr_ping_slot": ("time", np.ones(n, dtype=np.float64)),
            "sr_ping_group": ("time", np.ones(n, dtype=np.float64)),
            "sr_platform_state": ("time", np.full(n, 3, dtype=np.float64)),
        },
        coords={"time": time},
    )

    if include_positions:
        ds["depth"] = ("time", np.linspace(10, 50, n))
        ds["lat"] = ("time", np.linspace(44.0, 44.1, n))
        ds["lon"] = ("time", np.linspace(-124.0, -123.9, n))

    return ds


# ---------------------------------------------------------------------------
# Unit tests – write_riot_csv
# ---------------------------------------------------------------------------


class TestWriteRiotCsv:
    """Tests for write_riot_csv."""

    def test_creates_csv(self, tmp_path: object) -> None:
        """A valid dataset produces a CSV file."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = _make_riot_dataset()
        write_riot_csv(ds, add_positions=False, output_path=out)

        assert os.path.exists(out)

    def test_csv_columns_without_positions(self, tmp_path: object) -> None:
        """CSV should have the standard RIOT columns when positions are off."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        write_riot_csv(_make_riot_dataset(), add_positions=False, output_path=out)

        df = pd.read_csv(out)
        expected = [
            "riotDataPrefix",
            "epochMsecs",
            "rtMsecs",
            "freq",
            "detectionLevel",
            "sequenceNumber",
            "platformId",
            "slot",
            "group",
            "platformState",
        ]
        assert list(df.columns) == expected

    def test_csv_columns_with_positions(self, tmp_path: object) -> None:
        """When positions are enabled and available, depth/lat/lon columns appear."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = _make_riot_dataset(include_positions=True)
        write_riot_csv(ds, add_positions=True, output_path=out)

        df = pd.read_csv(out)
        for col in ("depth", "lat", "lon"):
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_columns_with_positions_missing_vars(self, tmp_path: object) -> None:
        """When positions are requested but vars are missing, blank columns appear."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = _make_riot_dataset(include_positions=False)
        write_riot_csv(ds, add_positions=True, output_path=out)

        df = pd.read_csv(out)
        for col in ("depth", "lat", "lon"):
            assert col in df.columns
            assert df[col].isna().all(), f"{col} should be all NaN"

    def test_row_count(self, tmp_path: object) -> None:
        """Output should have as many rows as valid (non-zero) time steps."""
        n = 8
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        write_riot_csv(_make_riot_dataset(n=n), add_positions=False, output_path=out)

        df = pd.read_csv(out)
        assert len(df) == n

    def test_epoch_msecs_calculation(self, tmp_path: object) -> None:
        """epochMsecs should equal days*86400000 + secs*1000 + msecs."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = _make_riot_dataset(n=3)
        write_riot_csv(ds, add_positions=False, output_path=out)

        df = pd.read_csv(out)
        days = ds["sr_ping_epoch_days"].values.astype(np.int64)
        secs = ds["sr_ping_secs"].values.astype(np.int64)
        msecs = ds["sr_ping_msecs"].values.astype(np.int64)
        expected = days * 86400 * 1000 + secs * 1000 + msecs
        np.testing.assert_array_equal(df["epochMsecs"].values, expected)

    def test_append_mode(self, tmp_path: object) -> None:
        """Calling write_riot_csv twice should append without repeating headers."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = _make_riot_dataset(n=3)
        write_riot_csv(ds, add_positions=False, output_path=out)
        write_riot_csv(ds, add_positions=False, output_path=out)

        with open(out) as f:
            lines = f.readlines()

        # header once + 3 rows + 3 appended rows = 7 lines
        assert len(lines) == 7

    def test_riot_data_prefix(self, tmp_path: object) -> None:
        """Every row should have '$riotData' in the riotDataPrefix column."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        write_riot_csv(_make_riot_dataset(), add_positions=False, output_path=out)

        df = pd.read_csv(out)
        assert (df["riotDataPrefix"] == "$riotData").all()

    def test_missing_variables_returns_early(self, tmp_path: object) -> None:
        """If required RIOT vars are missing, no file is created."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = xr.Dataset({"dummy": ("time", [1, 2, 3])})
        write_riot_csv(ds, add_positions=False, output_path=out)

        assert not os.path.exists(out)

    def test_all_zeros_dropped(self, tmp_path: object) -> None:
        """Rows where all RIOT variables are zero should be dropped."""
        out = str(tmp_path / "riot.csv")  # type: ignore[operator]
        ds = _make_riot_dataset(n=5)
        # Set all RIOT vars in the first row to 0
        for var in ds.data_vars:
            ds[var].values[0] = 0
        write_riot_csv(ds, add_positions=False, output_path=out)

        df = pd.read_csv(out)
        assert len(df) == 4  # first row should have been dropped
