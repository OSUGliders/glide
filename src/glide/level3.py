# Level 3 processing parsed the level 2 processed data
# Data are binned in height

import logging

import gsw
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from rich.progress import track

from . import convert as conv
from . import profiles as pfls

_log = logging.getLogger(__name__)


def parse_l2(l2_file: str) -> xr.Dataset:
    """Parse the level 2 data."""
    return xr.open_dataset(l2_file)


def get_profile_indexes(ds: xr.Dataset) -> NDArray:
    """Find the dive and climb indexes."""
    id = pfls.contiguous_regions(np.isfinite(ds.dive_id.values))
    ic = pfls.contiguous_regions(np.isfinite(ds.climb_id.values))
    idxs = np.vstack((ic, id))
    return idxs[np.argsort(idxs[:, 0]), :]


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
    ds = ds.drop_vars(drop_variables)

    binned_profiles = []
    state = []  # Dive / climb state
    profile_lat = []
    profile_lon = []
    profile_time = []
    profile_time_start = []
    profile_time_end = []

    for row in track(idxs):
        state.append(s.values[row[0]])
        profile = ds.isel(time=slice(row[0], row[1]))

        profile_time_start.append(profile.time[0].values)
        profile_time_end.append(profile.time[-1].values)

        idx_mid = len(profile.time) // 2
        time_mid = profile.time.values[idx_mid]
        profile_time.append(time_mid)
        profile_lat.append(profile.lat.sel(time=time_mid, method="nearest"))
        profile_lon.append(profile.lon.sel(time=time_mid, method="nearest"))

        # Some type juggling is required to properly bin time
        profile["i"] = ("time", np.arange(len(profile.time)))
        profile = profile.swap_dims({"time": "i"}).reset_coords("time")
        profile["time"] = ("i", profile.time.data.astype(float), profile.time.attrs)
        binned = profile.groupby_bins("depth", depth_bins).mean()
        binned["time"] = (
            "depth_bins",
            binned.time.data.astype("M8[ns]"),
            binned.time.attrs,
        )

        binned["depth_bins"] = [db.mid for db in binned.depth_bins.values]
        binned_profiles.append(binned)

    ds_binned = xr.concat(binned_profiles, dim="profile")

    # Swap some things around so that the plots look nice.
    depth_attrs = ds_binned.depth.attrs
    ds_binned = ds_binned.drop_vars("depth")
    ds_binned = ds_binned.rename({"depth_bins": "depth"})
    ds_binned["depth"].attrs = depth_attrs
    ds_binned["z"] = ("depth", -ds_binned.depth.values, z_attrs)
    ds_binned = ds_binned.swap_dims({"depth": "z"}).set_coords("z")

    ds_binned["state"] = (("profile",), state)
    ds_binned["profile_time"] = (("profile",), profile_time)
    ds_binned["profile_time_start"] = (("profile",), profile_time_start)
    ds_binned["profile_time_end"] = (("profile",), profile_time_end)
    ds_binned["profile_lat"] = (("profile",), profile_lat)
    ds_binned["profile_lon"] = (("profile",), profile_lon)

    ds_binned["profile_id"] = (("profile",), np.arange(1, len(ds_binned.profile) + 1))
    ds_binned = ds_binned.swap_dims({"profile": "profile_id"}).set_coords("profile_id")
    # ds_binned = ds_binned.set_coords(["profile_time", "profile_lat", "profile_lon"])

    return ds_binned.transpose()


def bin_q(ds: xr.Dataset, q_netcdf: str, bin_size: float, config: dict) -> xr.Dataset:
    _log.debug("Loading Q files")
    # Extract a subset of just the dissipation values
    qds = xr.open_mfdataset(q_netcdf, decode_timedelta=False)

    eds = qds[["e_1", "e_2"]]
    eds["depth"] = -gsw.z_from_p(qds.pressure, ds.profile_lat.mean().values)

    depth_bins = np.arange(
        -ds.z[0] - bin_size / 2, -ds.z[-1] + 1.5 * bin_size, bin_size
    )
    _log.debug("Epsilon depth bins %s", depth_bins)

    # Initialize data arrays
    dims = ds.conductivity.dims

    dissipation_variables = ["e_1", "e_2"]
    for v in track(dissipation_variables):
        ds[v] = (dims, np.full_like(ds.conductivity.values, np.nan), config[v]["CF"])
        # Convert from log
        eds[v] = (eds[v].dims, 10 ** eds[v].values)

    for i in range(ds.profile_id.size):
        ds_ = ds.isel(profile_id=i)
        eds_ = eds.sel(time=slice(ds_.profile_time_start, ds_.profile_time_end))
        if eds_.e_1.size < 1:
            _log.debug("No epsilon data")
            continue
        binned = eds_.groupby_bins("depth", depth_bins).mean()

        for v in dissipation_variables:
            ds[v][:, i] = binned[v].values

    return ds
