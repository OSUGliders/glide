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
