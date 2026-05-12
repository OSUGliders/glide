# gliderdac preparation functions split profiles into compliant files

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from .config import _load_core

_log = logging.getLogger(__name__)

_DROP_VARS: tuple[str, ...] = ("dive_id", "climb_id", "state")
_DROP_DIMS: tuple[str, ...] = ("time_gps",)
_SCALAR_FROM_TIME_UV: tuple[str, ...] = ("u", "v", "time_uv", "lat_uv", "lon_uv")
_INT_FILL = np.int32(-2147483647)


def _load_ngdac_config() -> dict:
    return _load_core()["ngdac"]


def _slice_profile(
    ds: xr.Dataset,
    prof_mask: np.ndarray,
    uv_idx: int,
    profile_id: int,
) -> xr.Dataset:
    scalar_values = {}
    for v in _SCALAR_FROM_TIME_UV:
        if v in ds:
            scalar_values[v] = (ds[v].values[uv_idx], dict(ds[v].attrs))

    pid_attrs = dict(ds["profile_id"].attrs) if "profile_id" in ds else {}

    prof_indices = np.where(prof_mask)[0]
    prof_ds = ds.isel(time=prof_indices)

    if "time_uv" in prof_ds.dims:
        prof_ds = prof_ds.drop_dims("time_uv")

    for v, (val, attrs) in scalar_values.items():
        prof_ds[v] = ((), val, attrs)

    if "profile_id" in prof_ds:
        prof_ds = prof_ds.drop_vars("profile_id")
    prof_ds["profile_id"] = ((), np.int32(profile_id), pid_attrs)

    for dim in _DROP_DIMS:
        if dim in prof_ds.dims:
            prof_ds = prof_ds.drop_dims(dim)

    for v in _DROP_VARS:
        if v in prof_ds:
            prof_ds = prof_ds.drop_vars(v)

    return prof_ds


def _add_ngdac_structural_vars(
    prof_ds: xr.Dataset,
    glider_name: str,
    instruments: dict | None,
    ngdac_cfg: dict,
) -> xr.Dataset:
    time_vals = prof_ds.time.values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        epoch_sec = time_vals.astype("datetime64[ns]").astype("int64") / 1e9
        profile_time = float(np.nanmean(epoch_sec))
    else:
        profile_time = float(np.nanmean(time_vals.astype("f8")))

    if "lat" in prof_ds:
        profile_lat = float(np.nanmean(prof_ds.lat.values.astype("f8")))
    else:
        profile_lat = float("nan")
    if "lon" in prof_ds:
        profile_lon = float(np.nanmean(prof_ds.lon.values.astype("f8")))
    else:
        profile_lon = float("nan")

    prof_ds["profile_time"] = ((), profile_time, ngdac_cfg["profile_time"]["CF"])
    prof_ds["profile_lat"] = ((), profile_lat, ngdac_cfg["profile_lat"]["CF"])
    prof_ds["profile_lon"] = ((), profile_lon, ngdac_cfg["profile_lon"]["CF"])

    instrument_list = ", ".join(instruments) if instruments else " "
    platform_attrs = {
        "_FillValue": _INT_FILL,
        **ngdac_cfg["platform"]["CF"],
        "id": glider_name,
        "instrument": instrument_list,
    }
    prof_ds["platform"] = ((), np.int32(0), platform_attrs)

    crs_attrs = {"_FillValue": _INT_FILL, **ngdac_cfg["crs"]["CF"]}
    prof_ds["crs"] = ((), np.int32(0), crs_attrs)

    for name, attrs in (instruments or {}).items():
        prof_ds[name] = ((), np.int32(0), {"_FillValue": _INT_FILL, **attrs})

    return prof_ds


def emit_ioos_profiles(
    ds: xr.Dataset,
    outdir: str | Path,
    glider_name: str,
    instruments: dict | None = None,
    force: bool = False,
    ngdac: dict | None = None,
) -> list[Path]:
    """Emit one IOOS NGDAC NetCDF file per profile."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for required in ("profile_id", "state"):
        if required not in ds:
            _log.warning("No %s in dataset, cannot emit IOOS files", required)
            return []
    if "time_uv" not in ds.dims:
        _log.warning("No time_uv dimension in dataset, cannot emit IOOS files")
        return []

    ngdac_cfg = ngdac or _load_ngdac_config()
    profile_ids = np.unique(ds.profile_id.values)
    profile_ids = profile_ids[profile_ids >= 0]

    time_uv_vals = ds.time_uv.values
    u_vals = ds.u.values
    v_vals = ds.v.values
    time_vals = ds.time.values
    state_vals = ds.state.values
    profile_id_vals = ds.profile_id.values
    n = len(state_vals)

    written: list[Path] = []
    for pid in profile_ids:
        pid = int(pid)
        prof_indices = np.where(profile_id_vals == pid)[0]
        if len(prof_indices) == 0:
            continue

        first_idx = int(prof_indices[0])
        last_idx = int(prof_indices[-1])
        left = first_idx
        while left > 0 and state_vals[left - 1] != 0:
            left -= 1
        right = last_idx
        while right < n - 1 and state_vals[right + 1] != 0:
            right += 1

        seg_time_min = time_vals[left]
        seg_time_max = time_vals[right]
        uv_match = (time_uv_vals >= seg_time_min) & (time_uv_vals <= seg_time_max)
        if not uv_match.any():
            _log.debug("Skipping profile_id %d: no matching time_uv entry", pid)
            continue
        uv_idx = int(np.where(uv_match)[0][0])

        u_val = u_vals[uv_idx]
        v_val = v_vals[uv_idx]
        if not (np.isfinite(u_val) and np.isfinite(v_val)):
            _log.debug(
                "Skipping profile_id %d: u/v not finite (segment awaiting closing surfacing)",
                pid,
            )
            continue

        prof_mask = profile_id_vals == pid
        prof_times = time_vals[prof_mask]
        first_time = float(prof_times[0])
        timestamp = datetime.fromtimestamp(first_time, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        out_path = outdir / f"{glider_name}_{timestamp}.nc"

        if out_path.exists() and not force:
            _log.debug("Skipping profile_id %d: %s already exists", pid, out_path.name)
            continue

        prof_ds = _slice_profile(ds, prof_mask, uv_idx, pid)
        prof_ds = _add_ngdac_structural_vars(
            prof_ds,
            glider_name,
            instruments,
            ngdac_cfg,
        )
        prof_ds.to_netcdf(out_path)
        written.append(out_path)
        _log.info(
            "Wrote %s (profile_id=%d, %d points)",
            out_path.name,
            pid,
            int(prof_mask.sum()),
        )

    _log.info("Emitted %d IOOS profile files to %s", len(written), outdir)
    return written
