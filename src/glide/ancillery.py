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


def parse_eps_bin(eps_file: str) -> xr.Dataset:
    """Load depth-binned microstructure produced by pyturb.

    Every variable is kept, unlike `parse_q`: `process_l3.merge_eps_bin` decides
    which ones to import from the `merged_variables` configuration. Times are
    decoded, because the per-bin `time` variable is what the profile alignment
    uses and it has to arrive as datetime64.
    """
    _log.debug("Loading microstructure eps-bin files")
    ds = xr.open_mfdataset(
        eps_file,
        concat_dim="profile",
        combine="nested",
        compat="override",
        coords="minimal",
        data_vars="minimal",
        decode_timedelta=False,
    ).load()

    missing = {"profile", "depth"} - {str(d) for d in ds.sizes}
    if missing:
        raise ValueError(
            f"{eps_file} does not look like a pyturb eps-bin file: missing "
            f"dimension(s) {sorted(missing)}, found {dict(ds.sizes)}."
        )

    # open_mfdataset does not record where the data came from, unlike
    # open_dataset, and the merge puts this in the L3 attributes.
    ds.encoding["source"] = eps_file

    _log.info(
        "Loaded %d microstructure profiles x %d depth bins",
        ds.sizes["profile"],
        ds.sizes["depth"],
    )
    return ds
