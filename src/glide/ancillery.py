# Functions for reading non-glide files
import logging

import xarray as xr

_log = logging.getLogger(__name__)


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


def parse_q(q_file: str) -> xr.Dataset:
    _log.debug("Loading Q files")
    return xr.open_mfdataset(q_file, decode_timedelta=False)[
        ["e_1", "e_2", "pressure"]
    ].load()
