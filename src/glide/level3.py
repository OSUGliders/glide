# Level 3 processing parsed the level 2 processed data
# Data are binned in height

import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray

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
    for row in idxs:
        state.append(s.values[row[0]])
        profile = ds.isel(time=slice(row[0], row[1]))
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

    return ds_binned.transpose()
