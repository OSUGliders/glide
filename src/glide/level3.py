# Level 3 processing parsed the level 2 processed data
# Data are binned in height

import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from rich.progress import track

from . import profiles as pfls

_log = logging.getLogger(__name__)


def parse_l2(l2_file: str) -> xr.Dataset:
    """Parse the level 2 data."""
    return xr.open_dataset(l2_file)


def get_profile_indexes(ds: xr.Dataset) -> NDArray:
    """Find the dive and climb indexes."""
    id = pfls.contiguous_regions(np.isfinite(ds.dive_id.values))
    ic = pfls.contiguous_regions(np.isfinite(ds.climb_id.values))
    idxs = np.empty((id.shape[0] + ic.shape[0], id.shape[1]), dtype=int)
    idxs[::2, :] = id
    idxs[1::2, :] = ic
    return idxs


def bin_l2(ds: xr.Dataset, bin_size: float = 10.0) -> xr.Dataset:
    """Depth bin size specified in meters."""

    # Generate bins
    depth_max = ds.depth.max()
    depth_bins = np.arange(0, depth_max + bin_size, bin_size)
    _log.debug(
        "First left bin edge %.1f, last right bin edge %.1f",
        depth_bins[0],
        depth_bins[-1],
    )

    idxs = get_profile_indexes(ds)

    s = ds.state.copy()
    z_attrs = ds.z.attrs

    # Drop some variables
    qc_variables = [v for v in ds.variables if "_qc" in str(v)]
    drop_variables = qc_variables + ["dive_id", "climb_id", "state", "z"]
    _log.warning("Binning QC flags is not supported, dropping %s", drop_variables)
    ds = ds.drop(drop_variables)

    binned_profiles = []
    state = []  # Dive / climb state
    profile_lat = []
    profile_lon = []
    profile_time = []

    for row in track(idxs):
        state.append(s.values[row[0]])
        profile = ds.isel(time=slice(row[0], row[1]))

        idx_mid = len(profile.time) // 2
        time_mid = profile.time.values[idx_mid]
        profile_time.append(time_mid)
        profile_lat.append(profile.lat.sel(time=time_mid, method="nearest"))
        profile_lon.append(profile.lon.sel(time=time_mid, method="nearest"))

        profile["i"] = ("time", np.arange(len(profile.time)))
        profile = profile.swap_dims({"time": "i"}).reset_coords("time")
        binned = profile.groupby_bins("depth", depth_bins).mean()
        binned["depth_bins"] = [db.mid for db in binned.depth_bins.values]
        binned_profiles.append(binned)

    ds_binned = xr.concat(binned_profiles, dim="profile")

    # Swap some things around so that the plots look nice.
    depth_attrs = ds_binned.depth.attrs
    ds_binned = ds_binned.drop("depth")
    ds_binned = ds_binned.rename({"depth_bins": "depth"})
    ds_binned["depth"].attrs = depth_attrs
    ds_binned["z"] = ("depth", -ds_binned.depth.values, z_attrs)
    ds_binned = ds_binned.swap_dims({"depth": "z"}).set_coords("z")

    ds_binned["state"] = (("profile",), state)
    ds_binned["profile_time"] = (("profile",), profile_time)
    ds_binned["profile_lat"] = (("profile",), profile_lat)
    ds_binned["profile_lon"] = (("profile",), profile_lon)

    ds_binned["profile_id"] = (("profile",), np.arange(1, len(ds_binned.profile) + 1))
    ds_binned = ds_binned.swap_dims({"profile": "profile_id"}).set_coords("profile_id")
    ds_binned = ds_binned.set_coords(["profile_time", "profile_lat", "profile_lon"])

    return ds_binned.transpose()
