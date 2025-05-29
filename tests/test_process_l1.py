from importlib import resources

import pandas as pd
import xarray as xr

import glide.process_l1 as pl1
from glide.config import load_config


def get_test_data(sn: str = "684", ftype: str = "sbd") -> xr.Dataset:
    return (
        pd.read_csv(str(resources.files("tests").joinpath(f"data/osu{sn}.{ftype}.csv")))
        .set_index("i")
        .to_xarray()
    )


def test_format_variables() -> None:
    config = load_config()
    sbd = pl1.format_variables(get_test_data("684", "sbd"), config)
    assert hasattr(sbd, "time")
    assert hasattr(sbd.time, "units")
    tbd = pl1.format_variables(get_test_data("684", "tbd"), config)
    assert hasattr(tbd, "time")
    assert hasattr(tbd.time, "units")


def test_parse_l1() -> None:
    config = load_config()
    pl1.parse_l1(get_test_data("684", "sbd"), config)
    pl1.parse_l1(get_test_data("684", "tbd"), config)
