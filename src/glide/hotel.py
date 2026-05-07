# Functions for generating hotel and gps files which are used in the MicroRider processing.

import xarray as xr
from scipy.io import savemat


def create_structure(ds: xr.Dataset) -> dict:
    # Convert times to POSIX timestamps (seconds since epoch, float, retaining fractional seconds)
    times_posix = ds["time"].values.astype("datetime64[ns]").astype("float64") / 1e9
    structure = {
        "ctd_temp": {
            "time": times_posix,
            "data": ds["temperature"].values,
        },
        "ctd_cond": {
            "time": times_posix,
            "data": ds["conductivity"].values,
        },
    }
    return structure


def save_hotel(hotel_struct: dict, out_file: str) -> None:
    savemat(out_file, hotel_struct)


def extract_gps(ds: xr.Dataset) -> xr.Dataset:
    return ds[["lat", "lon"]].dropna("time")


def extract_gps_fixes(ds: xr.Dataset) -> xr.Dataset:
    if "time_gps" not in ds.dims:
        raise ValueError(
            "Dataset has no time_gps dimension — was it produced by glide l2?"
        )
    fix_vars = [v for v in ds.data_vars if ds[v].dims == ("time_gps",)]
    return ds[fix_vars].set_index(time_gps="time_gps")
