# Additional processing of the l3 data, including the assimiation
# of other variables such as epsilon.
import logging

import gsw
import numpy as np
import xarray as xr

_log = logging.getLogger(__name__)

# Helper functions


def _infer_bin_size(ds: xr.Dataset) -> float:
    return (ds.z[0].z - ds.z[1]).values.item()


# Public functions


def concat(file_list: list[str], concat_dim: str = "time") -> xr.Dataset:
    _log.debug("Loading files")
    return xr.open_mfdataset(
        file_list,
        concat_dim=concat_dim,
        combine="nested",
        compat="override",
        coords="minimal",
        decode_timedelta=False,
        data_vars="minimal",
    ).load()


def parse_l3(l3_file: str) -> tuple[xr.Dataset, float]:
    ds = xr.open_dataset(l3_file, decode_timedelta=True).load()
    bin_size = _infer_bin_size(ds)
    ds.close()  # Will enable overwrite of existing l3 file.
    return ds, bin_size


def parse_q(q_file: str) -> xr.Dataset:
    _log.debug("Loading Q files")
    return xr.open_mfdataset(q_file, decode_timedelta=False)[
        ["e_1", "e_2", "pressure"]
    ].load()


def bin_q(
    ds: xr.Dataset, ds_q: xr.Dataset, bin_size: float, config: dict
) -> xr.Dataset:
    ds_q["depth"] = -gsw.z_from_p(ds_q.pressure, ds.profile_lat.mean().values)

    depth_bins = np.arange(
        -ds.z[0] - bin_size / 2, -ds.z[-1] + 1.5 * bin_size, bin_size
    )
    _log.debug("Epsilon depth bins %s", depth_bins)

    dims = ds.conductivity.dims

    dissipation_variables = ["e_1", "e_2"]
    for v in dissipation_variables:
        ds[v] = (
            dims,
            np.full_like(ds.conductivity.values, np.nan),
            config["variables"][v]["CF"],
        )
        # Dissipation rate is stored in the q file as the log10 of the value.
        # Convert it to the actual value.
        ds_q[v] = (ds_q[v].dims, 10 ** ds_q[v].values)

    for i in range(ds.profile_id.size):
        ds_ = ds.isel(profile_id=i)
        eds_ = ds_q.sel(
            # The type changing here is needed when the L2 data is binned just prior to
            # binning the q file data, because the binning operation stores
            # the start and end times as seconds since 1970-01-01T00:00:00. When merging q
            # data into L3 file directly the start and end times should already be datetimes
            # because xarray parses the epoch upon loading.
            time=slice(
                ds_.profile_time_start.astype("M8[s]"),
                ds_.profile_time_end.astype("M8[s]"),
            )
        )
        if eds_.e_1.size < 1:
            _log.debug("No epsilon data")
            continue
        binned = eds_.groupby_bins("depth", depth_bins).mean()

        for v in dissipation_variables:
            ds[v][:, i] = binned[v].values

    return ds
