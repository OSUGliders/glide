from importlib import resources

import pandas as pd
import xarray as xr

import glide.level2 as l2


def get_test_data(sn: str = "684", ftype: str = "sbd") -> xr.Dataset:
    return (
        pd.read_csv(str(resources.files("tests").joinpath(f"data/osu{sn}.{ftype}.csv")))
        .set_index("i")
        .to_xarray()
    )


def test_promote_time():
    ds = xr.Dataset(
        {
            "i": [1, 2, 3, 4],
            "foo": ("i", [1739052713, float("NaN"), 1739052715, 43324210293429]),
        }
    )
    ds = l2.promote_time(ds, "foo")
    assert "foo" in ds.coords
    assert ds.foo.size == 2
    assert list(ds.foo.values) == [1739052713, 1739052715]


def test_load_variable_specs() -> None:
    var_specs, source_map = l2.load_variable_specs()
    assert "m_present_time" in source_map
    assert source_map["m_present_time"] == "time"
    assert var_specs["time"]["CF"]["units"] == "seconds since 1970-01-01T00:00:00Z"


def test_format_variables() -> None:
    sbd = l2.format_variables(get_test_data("684", "sbd"))
    assert hasattr(sbd, "time")
    assert hasattr(sbd.time, "units")
    tbd = l2.format_variables(get_test_data("684", "tbd"))
    assert hasattr(tbd, "time")
    assert hasattr(tbd.time, "units")


def test_parse_l1() -> None:
    l2.parse_l1(get_test_data("684", "sbd"))
    l2.parse_l1(get_test_data("684", "tbd"))
