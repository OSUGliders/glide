import os
from importlib import resources

import pandas as pd
import xarray as xr

import glide.hotel as hotel


def create_test_dataset() -> xr.Dataset:
    times = pd.date_range(start="2023-01-01", periods=8, freq="h").to_numpy(
        dtype="M8[ns]"
    )

    temperature = [14.5, 15.2, 16.1, 15.8, 14.9, 15.0, 16.3, 15.7]
    conductivity = [34.0, 36.5, 35.2, 33.8, 37.1, 34.9, 36.0, 35.5]
    lat = [45.0, 45.1, 45.2, 45.3, 45.4, 45.5, 45.6, float("NaN")]
    lon = [-123.0, -123.1, -123.2, -123.3, -123.4, -123.5, -123.6, float("NaN")]

    ds = xr.Dataset(
        data_vars={
            "temperature": ("time", temperature),
            "conductivity": ("time", conductivity),
            "lat": ("time", lat),
            "lon": ("time", lon),
        },
        coords={"time": times},
    )

    return ds


def test_create_structure() -> None:
    ds = create_test_dataset()
    hotel_struct = hotel.create_structure(ds)

    assert type(hotel_struct) is dict

    required_keys = ["ctd_temp", "ctd_cond"]

    for key in required_keys:
        assert key in hotel_struct
        assert "time" in hotel_struct[key]
        assert "data" in hotel_struct[key]
        assert len(hotel_struct[key]["time"]) == len(ds["time"])
        assert len(hotel_struct[key]["data"]) == len(ds["time"])


def test_save_hotel(tmp_path) -> None:
    ds = create_test_dataset()
    hotel_struct = hotel.create_structure(ds)

    assert type(hotel_struct) is dict

    out_file = str(resources.files("tests").joinpath("data/synthetic.hotel.mat"))
    hotel.save_hotel(hotel_struct, out_file)

    assert os.path.exists(out_file)


def test_extract_gps() -> None:
    ds = create_test_dataset()

    gps = hotel.extract_gps(ds)

    assert set(gps.variables) == {"lat", "lon", "time"}
    assert gps.time.size == 7


def create_test_dataset_with_gps_fixes() -> xr.Dataset:
    import numpy as np

    times = pd.date_range("2023-01-01", periods=10, freq="h").to_numpy(dtype="M8[ns]")
    fix_times = pd.date_range("2023-01-01", periods=3, freq="3h").to_numpy(
        dtype="M8[ns]"
    )

    ds = xr.Dataset(
        data_vars={
            "lat": ("time", np.linspace(45.0, 45.9, 10)),
            "lon": ("time", np.linspace(-123.0, -123.9, 10)),
            "lat_gps": ("time_gps", [45.0, 45.3, 45.6]),
            "lon_gps": ("time_gps", [-123.0, -123.3, -123.6]),
            "hdop_gps": ("time_gps", [1.2, 1.5, float("nan")]),
        },
        coords={"time": times, "time_gps": fix_times},
    )
    return ds


def test_extract_gps_fixes_returns_fix_variables() -> None:
    ds = create_test_dataset_with_gps_fixes()
    fixes = hotel.extract_gps_fixes(ds)

    assert "lat_gps" in fixes
    assert "lon_gps" in fixes
    assert "hdop_gps" in fixes
    assert fixes.sizes["time_gps"] == 3
    # Variables on the main time dimension must not appear
    assert "lat" not in fixes
    assert "lon" not in fixes


def test_extract_gps_fixes_no_time_gps_raises() -> None:
    import pytest

    ds = create_test_dataset()
    with pytest.raises(ValueError, match="time_gps"):
        hotel.extract_gps_fixes(ds)
