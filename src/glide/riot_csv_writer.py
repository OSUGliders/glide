import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

_log = logging.getLogger(__name__)

# Timestamp difference threshold:
#   The limit that the riot timestamp must be from the glider timestamp
#   to be considered close enough to interpolate position variables.
TS_DIFF_THRESHOLD = 20  # seconds


def write_riot_csv(ds: xr.Dataset, add_positions: bool, output_path: str) -> None:
    """Write xarray Dataset to a RIOT-formatted CSV file.
    The output is a wide, record-oriented CSV (one row per ping) whose
    columns correspond to the fixed RIOT variables expected in a
    RIOT Data User manual file format.
    At a minimum, this function writes the following RIOT ping fields
    as individual columns:
    - ``sr_ping_epoch_days``
    - ``sr_ping_secs``
    - ``sr_ping_msecs``
    - ``sr_ping_rt_msecs``
    - ``sr_ping_freq``
    - ``sr_ping_detection_level``
    - ``sr_ping_sequence_number``
    - ``sr_ping_platform_id``
    - ``sr_ping_slot``
    - ``sr_ping_group``
    - ``sr_platform_state``
    - ``sr_num_records_in_file``
    If ``add_positions`` is True and the dataset contains them, the
    following position variables are also included as additional
    columns:
    - ``depth``
    - ``lat``
    - ``lon``
    The resulting CSV, containing one record per ping with these
    columns, is written to ``output_path``.
    """
    _log.debug(f"Gathering RIOT variables for CSV {output_path}")
    riot_vars = [
        "sr_ping_epoch_days",
        "sr_ping_secs",
        "sr_ping_msecs",
        "sr_ping_rt_msecs",
        "sr_ping_freq",
        "sr_ping_detection_level",
        "sr_ping_sequence_number",
        "sr_ping_platform_id",
        "sr_ping_slot",
        "sr_ping_group",
        "sr_platform_state",
    ]

    # Check that all required RIOT variables are present in the dataset
    if not set(riot_vars).issubset(set(ds.data_vars)):
        _log.error("Dataset is missing required RIOT variables")
        return

    # Drop any variables that are not needed for RIOT output
    vars_to_drop = set(ds.variables).difference(riot_vars)
    riot_ds = ds.drop_vars(vars_to_drop)
    if riot_ds.sizes == 0:
        _log.error("No RIOT data available to create the CSV")
        return

    # ToDo: this drop zeros section should be moved to processing L2
    #   for issue#32, but finish the riot_csv branch first
    # Drop any records with all zeros or NaNs
    temp_riot_array = riot_ds.to_array()
    rows_to_keep = np.logical_not(
        np.all(np.logical_or(np.isnan(temp_riot_array), temp_riot_array == 0), axis=0)
    )
    riot_ds = riot_ds.where(rows_to_keep, drop=True)
    if riot_ds.sizes["time"] == 0:
        _log.error("No RIOT data available to create the CSV")
        return

    # typecasting according to RIOT User data manual
    epoch_days = riot_ds["sr_ping_epoch_days"].values.astype(np.int64)
    secs = riot_ds["sr_ping_secs"].values.astype(np.int64)
    msecs = riot_ds["sr_ping_msecs"].values.astype(np.int64)
    # calculate the epoch time in milliseconds
    epoch_msecs = np.empty_like(epoch_days, dtype=np.int64)
    epoch_msecs[:] = epoch_days * 86400 * 1000 + secs * 1000 + msecs

    # converting everything to Int64 type makes it all integers but with
    # 'NA' as a missing value, which will fill in as blank in the CSV.
    riot_df = riot_ds.to_pandas().astype("Int64")
    assert isinstance(riot_df, pd.DataFrame), "Expected DataFrame from multi-var Dataset"

    # drop the columns used to create epoch_msecs
    riot_df = riot_df.drop(
        ["sr_ping_epoch_days", "sr_ping_secs", "sr_ping_msecs"], axis=1
    )

    # rename columns to match headers in RIOT Data User Manual
    riot_df.columns = [
        "rtMsecs",
        "freq",
        "detectionLevel",
        "sequenceNumber",
        "platformId",
        "slot",
        "group",
        "platformState",
    ]

    # Add the additional columns
    riot_df.insert(loc=0, column="epochMsecs", value=epoch_msecs)
    riot_df.insert(loc=0, column="riotDataPrefix", value="$riotData")
    # riot_df['recNumInFile'] = np.nan  # unused record number in file.

    if add_positions:
        riot_df = _add_positions(ds, riot_df, rows_to_keep)

    # Write to CSV
    _log.debug("Writing to RIOT CSV")
    # If the file exists already, it will append, so don't write
    # the header.
    if os.path.exists(output_path):
        headerwrite = False
    else:
        headerwrite = True

    riot_df.to_csv(
        output_path, index=False, header=headerwrite, lineterminator="\n", mode="a"
    )


def _add_positions(ds, riot_df, rows_to_keep):
    """Add position variables (depth, lat, lon) to the RIOT DataFrame by
    interpolating from the glider position data to the RIOT
    timestamps.  Only RIOT timestamps that fall within the time
    boundaries of the available glider position data will be
    interpolated; others will be left as NaN (and thus blank in the
     CSV).
    """
    _log.debug("Adding position variables to RIOT CSV")
    # including depth in positions assumes that the thermodynamic
    # calculations were added.
    position_vars = ["depth", "lat", "lon", "time"]

    if not set(position_vars).issubset(ds.variables):
        missing_vars = set(position_vars).difference(ds.variables)
        _log.warning(
            f"Position variables {missing_vars} are missing from "
            "dataset, positions cannot be added to RIOT CSV filling "
            "with blanks"
        )
        riot_df["depth"] = np.nan
        riot_df["lat"] = np.nan
        riot_df["lon"] = np.nan
        return riot_df

    vars_to_drop = set(ds.variables).difference(position_vars)
    position_ds = ds.drop_vars(vars_to_drop)
    position_ds = position_ds.where(rows_to_keep, drop=True)

    # Gather the timestamps for checking if interpolation is possible
    riot_ts = riot_df["epochMsecs"] / 1000
    glider_ts = position_ds["time"].values

    # pre-allocate arrays with NaNs
    depth = np.full(riot_ts.shape, np.nan)
    lat = np.full(riot_ts.shape, np.nan)
    lon = np.full(riot_ts.shape, np.nan)

    q_depth = np.logical_and(
        np.isfinite(position_ds["depth"]), position_ds["depth"] != 0
    )
    q_pos = np.logical_and(
        np.isfinite(position_ds["lat"]), np.isfinite(position_ds["lon"])
    )

    if np.sum(q_depth) == 0:
        _log.warning("No valid depths found. adding blank positions")
        riot_df["depth"] = depth
        riot_df["lat"] = lat
        riot_df["lon"] = lon
        return riot_df

    _log.debug(
        "Interpolating position variables to RIOT timestamps that fall "
        "within the glider position time boundaries"
    )

    # Only interpolate to timestamps that fall within the time boundaries
    # of the available glider position data. Any that fall outside
    # will be NaNs and ultimately blanks in the CSV file.
    qdepth_in_tbnds = np.logical_and(
        riot_ts >= glider_ts[q_depth][0], riot_ts <= glider_ts[q_depth][-1]
    )
    qpos_in_tbnds = np.logical_and(
        riot_ts >= glider_ts[q_pos][0], riot_ts <= glider_ts[q_pos][-1]
    )

    depth[qdepth_in_tbnds] = np.interp(
        riot_ts[qdepth_in_tbnds], glider_ts[q_depth], position_ds["depth"][q_depth]
    )
    lat[qpos_in_tbnds] = np.interp(
        riot_ts[qpos_in_tbnds], glider_ts[q_pos], position_ds["lat"][q_pos]
    )
    lon[qpos_in_tbnds] = np.interp(
        riot_ts[qpos_in_tbnds], glider_ts[q_pos], position_ds["lon"][q_pos]
    )

    riot_df["depth"] = depth
    riot_df["lat"] = lat
    riot_df["lon"] = lon

    return riot_df
