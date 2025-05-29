# Level 3 processing parsed the level 2 processed data
# Data are binned in depth

import logging

import gsw
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from . import profiles as pfls

_log = logging.getLogger(__name__)


def parse_l2(l2_file: str) -> xr.Dataset:
    """Parse the level 2 data."""
    return xr.open_dataset(l2_file).load()


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

    # Drop some variables but same some attributes
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
    state = []  # Dive / climb state
    profile_lat = []
    profile_lon = []
    profile_time = []
    profile_time_start = []
    profile_time_end = []

    ds["time"] = ds.time.astype("f8") / 1e9  # Convert to seconds posix
    time_attrs["units"] = "seconds since 1970-01-01T00:00:00"

    for row in idxs:
        state.append(s.values[row[0]])
        profile = ds.isel(time=slice(row[0], row[1]))

        # Store profile timing and position
        time = profile.time.values
        profile_time_start.append(time[0])
        profile_time_end.append(time[-1])
        idx_mid = np.searchsorted(time, 0.5 * (time[0] + time[-1]))
        profile_time.append(time[idx_mid])
        profile_lat.append(profile.lat.values[idx_mid])
        profile_lon.append(profile.lon.values[idx_mid])

        # Some juggling is required to properly bin time
        # Make a new variable i
        profile["i"] = ("time", np.arange(len(profile.time)))
        # Make i the coordinate and swap time to a variable
        profile = profile.swap_dims({"time": "i"}).reset_coords("time")
        profile["time"].attrs = time_attrs
        binned = profile.groupby_bins("depth", depth_bins).mean()

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

    ds_binned["state"] = (
        ("profile",),
        state,
        dict(
            long_name="Glider state",
            flag_values=np.array([1, 2]).astype("i1"),
            flag_meanings="diving climbing",
        ),
    )
    ds_binned["profile_time"] = (("profile",), profile_time, time_attrs)
    ds_binned["profile_time_start"] = (("profile",), profile_time_start, time_attrs)
    ds_binned["profile_time_end"] = (("profile",), profile_time_end, time_attrs)
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
    for v in dissipation_variables:
        ds[v] = (
            dims,
            np.full_like(ds.conductivity.values, np.nan),
            config["variables"][v]["CF"],
        )
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
