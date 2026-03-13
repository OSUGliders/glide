import logging

import numpy as np
import pandas as pd
import xarray as xr

_log = logging.getLogger(__name__)

# Timestamp difference threshold:
#   The limit that the riot timestamp must be from the glider timestamp
#   to be considered close enough to interpolate position variables.
TS_DIFF_THRESHOLD = 20  # seconds


def write_riot_csv(ds: xr.Dataset, add_positions: bool, output_path: str) -> None:
    """Write xarray Dataset to a RIOT `$riotData`-style CSV file.
    The output is a wide, record-oriented CSV (one row per ping) whose
    columns correspond to the fixed RIOT variables expected in a
    `$riotData` file format.
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

    # including depth in positions assumes that the thermodynamic
    # calculations were added.
    position_vars = ["depth", "lat", "lon", "time"]

    # Check that all required RIOT variables are present in the dataset
    if not set(riot_vars).issubset(set(ds.data_vars)):
        _log.error("Dataset is missing required RIOT variables")
        return

    # Optionally add position variables if they are present in the dataset
    if add_positions and not set(position_vars).issubset(ds.variables):
        missing_vars = set(position_vars).difference(ds.variables)
        _log.warning(
            f"Position variables {missing_vars} are missing from "
            "dataset, positions cannot be added to RIOT CSV"
        )
        add_positions = False

    # Drop any variables that are not needed for RIOT output
    vars_to_drop = set(ds.data_vars).difference(riot_vars)
    riot_ds = ds.drop_vars(vars_to_drop)

    # ToDo: this drop zeros section should be moved to processing L2
    #   for issue#32, but finish the riot_csv branch first
    # Drop any records with all zeros or NaNs
    temp_riot_array = riot_ds.to_array()
    rows_to_keep = np.logical_not(
        np.all(np.logical_or(np.isnan(temp_riot_array), temp_riot_array == 0), axis=0)
    )
    riot_ds = riot_ds.where(rows_to_keep, drop=True)

    # typecasting according to RIOT User data manual
    epoch_days = riot_ds["sr_ping_epoch_days"].values.astype(np.int64)
    secs = riot_ds["sr_ping_secs"].values.astype(np.int64)
    msecs = riot_ds["sr_ping_msecs"].values.astype(np.int64)
    # calculate the epoch time in milliseconds
    epoch_msecs = np.empty_like(epoch_days, dtype=np.int64)
    epoch_msecs[:] = epoch_days * 86400 * 1000 + secs * 1000 + msecs

    rt_msecs = riot_ds["sr_ping_rt_msecs"].values.astype(np.uint32)
    freq = riot_ds["sr_ping_freq"].values.astype(np.uint32)
    detection_level = riot_ds["sr_ping_detection_level"].values.astype(np.uint16)
    sequence_number = riot_ds["sr_ping_sequence_number"].values.astype(np.uint32)
    platform_id = riot_ds["sr_ping_platform_id"].values.astype(np.uint8)
    slot = riot_ds["sr_ping_slot"].values.astype(np.uint8)
    group = riot_ds["sr_ping_group"].values.astype(np.uint8)
    platform_state = riot_ds["sr_platform_state"].values.astype(np.int32)

    riot_df = pd.DataFrame(
        {
            "riotData_prefix": np.full(len(epoch_msecs), "$riotData"),
            "epoch_msecs": epoch_msecs,
            "rt_msecs": rt_msecs,
            "freq": freq,
            "detection_level": detection_level,
            "sequence_number": sequence_number,
            "platform_id": platform_id,
            "slot": slot,
            "group": group,
            "state": platform_state,
            "num_records": np.full(len(epoch_msecs), np.nan),
        }
    )

    if add_positions:
        _log.debug("Adding position variables to RIOT CSV")
        vars_to_drop = set(ds.data_vars).difference(position_vars)
        position_ds = ds.drop_vars(vars_to_drop)
        position_ds = position_ds.where(rows_to_keep, drop=True)
        riot_ts = epoch_msecs / 1000
        glider_ts = position_ds["time"]
        # ToDo: potentially move this section to _interpolate ...
        if np.all(abs(riot_ts - glider_ts) < TS_DIFF_THRESHOLD):
            _log.debug("Interpolating position variables to RIOT timestamps")
            riot_df["depth"] = np.interp(riot_ts, glider_ts, position_ds["depth"])
            riot_df["lat"] = np.interp(riot_ts, glider_ts, position_ds["lat"])
            riot_df["lon"] = np.interp(riot_ts, glider_ts, position_ds["lon"])
        else:
            _log.warning(
                f"RIOT timestamps greater than "
                f"Threshold:{TS_DIFF_THRESHOLD} from glider timestamps. "
                f"Using record's coordinates instead of interpolating."
            )
            riot_df["depth"] = position_ds["depth"]
            riot_df["lat"] = position_ds["lat"]
            riot_df["lon"] = position_ds["lon"]

    # Write to CSV
    _log.debug("Writing to RIOT CSV")
    riot_df.to_csv(
        output_path, index=False, header=False, lineterminator="\n", mode="a"
    )
