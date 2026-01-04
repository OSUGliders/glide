# Level 3 processing of the level 2 data
# Data are binned in depth

import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

# Helper functions


def _contiguous_regions(condition: NDArray) -> NDArray:
    """Finds the indices of contiguous True regions in a boolean array.

    Parameters
    ----------
    condition : array_like
            Array of boolean values.

    Returns
    -------
    idx : ndarray
            Array of indices demarking the start and end of contiguous True regions in condition.
            Shape is (N, 2) where N is the number of regions.

    """

    condition = np.asarray(condition)
    d = np.diff(condition)
    (idx,) = d.nonzero()
    idx += 1

    if condition[0]:
        idx = np.r_[0, idx]

    if condition[-1]:
        idx = np.r_[idx, condition.size]

    idx.shape = (-1, 2)
    return idx


def _get_profile_indexes(ds: xr.Dataset) -> NDArray:
    """Find the dive and climb indexes."""
    dives = _contiguous_regions(np.isfinite(ds.dive_id.values))
    climbs = _contiguous_regions(np.isfinite(ds.climb_id.values))
    idxs = np.vstack((climbs, dives))
    return idxs[np.argsort(idxs[:, 0]), :]


# Public functions


def parse_l2(file: str) -> xr.Dataset:
    return xr.open_dataset(file, decode_timedelta=True).load()


def bin_l2(
    ds: xr.Dataset, bin_size: float = 10.0, depth: float | None = None
) -> xr.Dataset:
    """Depth bin size specified in meters."""

    if depth is None:
        depth = ds.depth.max()
        _log.debug("Inferring max depth from data %.1f", depth)

    depth_bins = np.arange(0, depth + bin_size, bin_size)
    _log.debug(
        "First left bin edge %.1f, last right bin edge %.1f",
        depth_bins[0],
        depth_bins[-1],
    )

    profile_indexes = _get_profile_indexes(ds)

    # Dropping QC variables for now because it isn't clear how we should
    # treat them after binning. We may want to define new QC variables in the future.
    qc_variables = [v for v in ds.variables if "_qc" in str(v)]
    drop_variables = qc_variables + ["dive_id", "climb_id", "state", "z"]
    _log.warning("Binning QC flags is not supported, dropping %s", drop_variables)
    for var in ds.variables:
        if "ancillary_variables" not in ds[var].attrs:
            continue
        if ds[var].attrs["ancillary_variables"] in qc_variables:
            ds[var].attrs.pop("ancillary_variables")

    s = ds.state.copy()
    z_attrs = ds.z.attrs.copy()
    time_attrs = ds.time.attrs.copy()

    ds = ds.drop_vars(drop_variables)

    binned_profiles = []
    state = []
    profile_lat = []
    profile_lon = []
    profile_time = []
    profile_time_start = []
    profile_time_end = []

    # To properly bin time we have to make sure it is a floating point variable.
    # So we need to make a new placeholder coordinate
    ds["time"] = (
        ds.time.astype("f8") / 1e9
    )  # Convert to seconds posix. 1e9 is needed otherwise we get nanoseconds.
    time_attrs["units"] = "seconds since 1970-01-01T00:00:00"
    ds["time"].attrs = time_attrs
    ds["i"] = ("time", np.arange(len(ds.time)))
    ds = ds.swap_dims({"time": "i"}).reset_coords("time")

    for row in profile_indexes:
        profile = ds.isel(i=slice(row[0], row[1]))
        time = profile.time.values
        idx_mid = np.searchsorted(time, 0.5 * (time[0] + time[-1]))

        state.append(s.values[row[0]])
        profile_time_start.append(time[0])
        profile_time_end.append(time[-1])
        profile_time.append(time[idx_mid])
        profile_lat.append(profile.lat.values[idx_mid])
        profile_lon.append(profile.lon.values[idx_mid])

        binned = profile.groupby_bins("depth", depth_bins).mean()

        binned["depth_bins"] = [db.mid for db in binned.depth_bins.values]
        binned_profiles.append(binned)

    _log.debug("Concatenating %i binned profiles", len(binned_profiles))
    ds_binned = xr.concat(binned_profiles, dim="profile")

    # This rearrangement of depth and height makes xarray plots look better.
    depth_attrs = ds_binned.depth.attrs
    ds_binned = ds_binned.drop_vars("depth")
    ds_binned = ds_binned.rename({"depth_bins": "depth"})
    ds_binned["depth"].attrs = depth_attrs
    ds_binned["z"] = ("depth", -ds_binned.depth.values, z_attrs)
    ds_binned = ds_binned.swap_dims({"depth": "z"}).set_coords("z")

    ds_binned["state"] = (
        ("profile",),
        state,
        dict(
            long_name="Glider state",
            flag_values=np.array([1, 2]).astype("i1"),
            flag_meanings="dive climb",
        ),
    )
    ds_binned["profile_time"] = (("profile",), profile_time, time_attrs)
    ds_binned["profile_time_start"] = (("profile",), profile_time_start, time_attrs)
    ds_binned["profile_time_end"] = (("profile",), profile_time_end, time_attrs)
    ds_binned["profile_lat"] = (("profile",), profile_lat)
    ds_binned["profile_lon"] = (("profile",), profile_lon)

    ds_binned["profile_id"] = (("profile",), np.arange(1, len(ds_binned.profile) + 1))
    ds_binned = ds_binned.swap_dims({"profile": "profile_id"}).set_coords("profile_id")
    ds_binned = ds_binned.set_coords(["profile_time", "profile_lat", "profile_lon"])

    return ds_binned.transpose()
