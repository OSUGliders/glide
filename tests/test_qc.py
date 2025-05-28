import numpy as np
import xarray as xr

import glide.qc as qc


def initialise_test_data() -> xr.Dataset:
    # Fake data with 3 surfaces and 2 profiles inbetween.
    # Four points in each surfacing. Five points in the profiles.
    ds = xr.Dataset(
        {
            "lon": (
                "time",
                [
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    122.2,
                    122.3,
                    122.4,
                    122.5,
                    122.6,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    123.2,
                    123.7,
                    124,
                    124.5,
                    124.6,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                ],
                {
                    "standard_name": "longitude",
                    "long_name": "Longitude",
                },
            ),
            "lat": (
                "time",
                [
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    22.2,
                    22.3,
                    5000,
                    22.5,
                    22.6,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    23.2,
                    23.7,
                    24,
                    24.5,
                    24.6,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                ],
            ),
            "m_gps_lon": (
                "time",
                [
                    122,
                    122,
                    122,
                    122,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    123,
                    123,
                    123,
                    123,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    125,
                    float("NaN"),
                    125,
                    125,
                ],
            ),
            "m_gps_lat": (
                "time",
                [
                    22,
                    22,
                    22,
                    22,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    23,
                    23,
                    23,
                    23,
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    25,
                    25,
                    25,
                    25,
                ],
            ),
            "m_depth": (
                "time",
                [0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0],
            ),
        },
        {
            "time": (
                "time",
                [
                    0,
                    4,
                    8,
                    12,
                    16,
                    116,
                    216,
                    316,
                    416,
                    516,
                    520,
                    524,
                    530,
                    534,
                    634,
                    734,
                    834,
                    934,
                    1038,
                    1042,
                    1046,
                    1050,
                ],
            )
        },
    )
    return ds


def test_fit_line() -> None:
    assert np.isclose(qc.fit_line(0, 0, 1, 2), (2, 0)).all()
    assert np.isclose(qc.fit_line(-2, 2, 2, -1), (-3 / 4, 2 - 3 / 2)).all()


def test_nan_out_of_bounds() -> None:
    y = [0.0, 1.1, 2.0, 2.9, 4.0]
    y_ = qc.nan_out_of_bounds(y, 1, 3)
    assert np.isnan(y_[0])
    assert np.isnan(y_[-1])
    assert np.isfinite(y_[1:4]).all()


def test_time() -> None:
    ds = xr.Dataset(
        {
            "foo": (
                "i",
                [1739052713, 1739052713, float("NaN"), 1739052715, 43324210293429],
                dict(valid_min=946684800, valid_max=2208988800),
            ),
        }
    )
    ds = qc.time(ds, "foo")
    assert ds.foo.size == 3
    assert list(ds.foo.values) == [1739052713, 1739052715, 43324210293429]


def test_gps() -> None:
    ds = initialise_test_data()

    m0, c0 = qc.fit_line(
        ds.time[3].values,
        ds.m_gps_lon[3].values,
        ds.time[9].values,
        ds.m_gps_lon[9].values,
    )
    m, c = qc.fit_line(
        ds.time[4].values, ds.lon[4].values, ds.time[8].values, ds.lon[8].values
    )
    dl = (m0 - m) * ds.time[4:9].values + (c0 - c)
    lon_corrected = ds.lon[4:9].values + dl

    ds_ = qc.gps(ds.copy(), dt=50)

    assert np.isclose(ds_.lon[4:9].values, lon_corrected).all()


def test_init_qc_variable() -> None:
    ds = initialise_test_data()

    ds = qc.init_qc_variable(ds, "lon")
    assert "lon_qc" in ds.variables
    assert ds.lon_qc.attrs["standard_name"] == "longitude status_flag"
    assert ds.lon.attrs["ancillary_variables"] == "lon_qc"
    assert (ds.lon_qc[np.isnan(ds.lon_qc)] == 9).all()
