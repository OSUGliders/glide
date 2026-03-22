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


def _interp_velocity_to_profiles(
    ds_binned: xr.Dataset,
    profile_time: list,
    has_velocity: bool,
    vel_time: NDArray | None,
    vel_u: NDArray | None,
    vel_v: NDArray | None,
    u_attrs: dict,
    v_attrs: dict,
) -> xr.Dataset:
    """Interpolate depth-averaged velocity onto profile mid-point times.

    NaN velocity values propagate naturally: any profile whose bracketing
    velocity entries include a NaN will receive NaN. Profiles outside the
    time range of velocity reports also receive NaN (no extrapolation).

    Parameters
    ----------
    ds_binned : xr.Dataset
        Binned L3 dataset with profile_id dimension.
    profile_time : list
        Profile mid-point times in posix seconds.
    has_velocity : bool
        Whether velocity data exists in the L2 source.
    vel_time, vel_u, vel_v : array or None
        Velocity data; may contain NaN values which propagate to output.
    u_attrs, v_attrs : dict
        CF attributes for u and v variables.
    """
    n_profiles = ds_binned.profile_id.size
    profile_t = np.asarray(profile_time, dtype="f8")

    if (
        not has_velocity
        or vel_time is None
        or vel_u is None
        or vel_v is None
        or len(vel_time) < 1
    ):
        _log.info("No valid velocity data for L3 interpolation")
        ds_binned["u"] = (("profile_id",), np.full(n_profiles, np.nan), u_attrs)
        ds_binned["v"] = (("profile_id",), np.full(n_profiles, np.nan), v_attrs)
        return ds_binned

    # Interpolate using only the valid (finite) velocity points.
    valid = np.isfinite(vel_u) & np.isfinite(vel_v)
    if not valid.any():
        _log.info("All velocity values are NaN")
        ds_binned["u"] = (("profile_id",), np.full(n_profiles, np.nan), u_attrs)
        ds_binned["v"] = (("profile_id",), np.full(n_profiles, np.nan), v_attrs)
        return ds_binned

    u_interp = np.interp(
        profile_t, vel_time[valid], vel_u[valid], left=np.nan, right=np.nan
    )
    v_interp = np.interp(
        profile_t, vel_time[valid], vel_v[valid], left=np.nan, right=np.nan
    )

    # Propagate NaN: mask profiles whose bracketing velocity entries have NaN.
    # For each profile time, find the surrounding entries in the full
    # (including NaN) velocity array.  If either neighbour is NaN, the
    # interpolated value is set to NaN.
    for i, t in enumerate(profile_t):
        idx = np.searchsorted(vel_time, t)
        left_nan = idx > 0 and not np.isfinite(vel_u[idx - 1])
        right_nan = idx < len(vel_time) and not np.isfinite(vel_u[idx])
        if left_nan or right_nan:
            u_interp[i] = np.nan
            v_interp[i] = np.nan

    n_valid = np.isfinite(u_interp).sum()
    _log.info(
        "Interpolated velocity onto %d profiles (%d valid)",
        n_profiles,
        n_valid,
    )

    ds_binned["u"] = (("profile_id",), u_interp, u_attrs)
    ds_binned["v"] = (("profile_id",), v_interp, v_attrs)
    return ds_binned


# Public functions


def parse_l2(file: str) -> xr.Dataset:
    return xr.open_dataset(file, decode_timedelta=True).load()


def bin_l2(
    ds: xr.Dataset,
    bin_size: float = 10.0,
    depth: float | None = None,
    config: dict | None = None,
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
    velocity_variables = [
        v
        for v in ["u", "v", "lat_uv", "lon_uv", "time_uv"]
        if v in ds.variables or v in ds.dims
    ]
    drop_variables = (
        qc_variables + ["dive_id", "climb_id", "state", "z"] + velocity_variables
    )

    # Extract velocity data before dropping, for later interpolation onto profiles.
    # NaN velocity values are preserved so they propagate through interpolation.
    has_velocity = "time_uv" in ds.dims and "u" in ds
    if has_velocity:
        vel_time = ds.time_uv.values.astype("f8") / 1e9  # posix seconds
        vel_u = ds.u.values.astype("f8")
        vel_v = ds.v.values.astype("f8")
        # Only drop entries with invalid times; keep NaN u/v so they propagate.
        valid_time = np.isfinite(vel_time)
        vel_time = vel_time[valid_time]
        vel_u = vel_u[valid_time]
        vel_v = vel_v[valid_time]
        u_attrs = ds.u.attrs.copy()
        v_attrs = ds.v.attrs.copy()

    # Drop variables flagged with drop_from_l3 in the config
    if config is not None:
        for v in list(ds.variables):
            try:
                if config["variables"][v].get("drop_from_l3", False):
                    _log.info("Not binning %s due to drop_from_l3 flag in config", v)
                    drop_variables.append(v)
            except KeyError:
                pass
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

    # Interpolate velocity onto profile mid-point times.
    ds_binned = _interp_velocity_to_profiles(
        ds_binned,
        profile_time,
        has_velocity,
        vel_time if has_velocity else None,
        vel_u if has_velocity else None,
        vel_v if has_velocity else None,
        u_attrs if has_velocity else {},
        v_attrs if has_velocity else {},
    )

    return ds_binned.transpose()
