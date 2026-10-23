# Additional processing of the l3 data, including the assimilation
# of other variables such as epsilon.
import logging

import gsw
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from . import qc

_log = logging.getLogger(__name__)

_TOL = 1e-6  # m or s, tolerance for matching bin geometry

# Helper functions


def _infer_bin_size(ds: xr.Dataset) -> float:
    return (ds.z[0].z - ds.z[1]).values.item()


def _to_datetime64(values: ArrayLike) -> NDArray:
    """Coerce profile times to datetime64[ns].

    `bin_l2` leaves the profile start and end times as float seconds since the
    epoch, while reading an L3 file back decodes them to datetimes. Nanoseconds
    rather than seconds because a second of truncation can move a microstructure
    bin across a profile boundary.
    """
    a = np.asarray(values)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("M8[ns]")
    if np.issubdtype(a.dtype, np.number):
        return (np.asarray(a, dtype="f8") * 1e9).astype("i8").astype("M8[ns]")
    raise TypeError(f"Cannot interpret {a.dtype} as a time")


def _eps_bin_size(ds_eps: xr.Dataset) -> float:
    """Bin width of the microstructure depth grid, validated for alignment.

    pyturb records no bin width, so it has to come from the coordinate spacing.
    The grid must be uniform and start at the surface, because glide's L3 bins
    always do; that is what makes the two grids comparable at all.
    """
    depth = np.asarray(ds_eps["depth"].values, dtype="f8")
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError(
            f"Microstructure depth coordinate must be one dimensional with at "
            f"least two bins, got shape {depth.shape}."
        )

    spacing = np.diff(depth)
    if not np.all(spacing > 0):
        raise ValueError("Microstructure depth coordinate is not increasing.")
    if not np.allclose(spacing, spacing[0], rtol=0, atol=_TOL):
        raise ValueError(
            f"Microstructure depth bins are not uniformly spaced (spacings "
            f"{np.unique(np.round(spacing, 6))}); cannot infer a bin width."
        )

    width = float(spacing[0])
    if abs(depth[0] - width / 2) > _TOL:
        raise ValueError(
            f"Microstructure depth bins do not start at the surface: the first "
            f"centre is {depth[0]:g} m for a {width:g} m bin, implying a first "
            f"edge at {depth[0] - width / 2:g} m. glide L3 bins always start at "
            f"0 m, so the two grids cannot be aligned."
        )
    return width


def _check_bin_size(bin_size: float, eps_bin_size: float) -> None:
    """Refuse to merge microstructure onto a grid it was not binned on."""
    if abs(bin_size - eps_bin_size) > _TOL:
        raise ValueError(
            f"L3 depth bins are {bin_size:g} m but the microstructure file is "
            f"binned at {eps_bin_size:g} m. pyturb eps-bin data are already "
            f"depth-averaged, so glide will not re-bin them. Re-run 'glide l3' "
            f"with -b {eps_bin_size:g}, or merge into an L3 file that already "
            f"uses {eps_bin_size:g} m bins."
        )


def _depth_map(l3_centres: NDArray, eps_centres: NDArray) -> tuple[NDArray, NDArray]:
    """Index pairs matching microstructure depth bins to L3 depth bins.

    Returns the microstructure bin indices that have a home in the L3 grid and the
    L3 indices they map onto. Bins deeper than the L3 grid, or falling in a gap in
    it, are left out: `bin_l2` builds its depth coordinate from the bins that
    actually hold data, so the grid is not guaranteed to be contiguous.
    """
    pos = np.clip(np.searchsorted(l3_centres, eps_centres), 0, l3_centres.size - 1)
    lower = np.maximum(pos - 1, 0)
    closer = np.abs(l3_centres[lower] - eps_centres) < np.abs(
        l3_centres[pos] - eps_centres
    )
    pos = np.where(closer, lower, pos)

    matched = np.abs(l3_centres[pos] - eps_centres) < _TOL
    dropped = int((~matched).sum())
    if dropped:
        _log.info(
            "Dropped %d of %d microstructure depth bins outside the L3 depth grid "
            "(deeper than %.1f m, or in a gap in it).",
            dropped,
            matched.size,
            l3_centres.max(),
        )
    return np.flatnonzero(matched), pos[matched]


def _assign_eps_profiles(ds: xr.Dataset, ds_eps: xr.Dataset) -> tuple[NDArray, NDArray]:
    """Match each microstructure profile to the L3 profile it overlaps most.

    Every microstructure bin carries its own time, so a profile goes to the L3
    profile whose [profile_time_start, profile_time_end] window holds the most of
    those times. The per-profile `profile_time` is deliberately not used: it marks
    the start of the microstructure record, which begins while the glider is still
    descending, so it points at the preceding dive most of the time.

    Returns the index of the assigned L3 profile for each microstructure profile
    (-1 where none overlaps) and the fraction of its bin times that fell inside.
    """
    starts = _to_datetime64(ds["profile_time_start"].values)
    ends = _to_datetime64(ds["profile_time_end"].values)
    if not np.all(np.diff(starts) > np.timedelta64(0, "ns")):
        raise ValueError(
            "L3 profile start times are not increasing; cannot align microstructure."
        )

    times = np.asarray(ds_eps["time"].values).astype("M8[ns]")
    n_eps, n_l3 = times.shape[0], starts.size

    valid = ~np.isnat(times)
    flat = times[valid]
    owner = np.repeat(np.arange(n_eps), valid.sum(axis=1))

    # The last profile that started at or before each bin time. Profile windows
    # can overlap by a few seconds, and taking the later start counts each bin
    # time exactly once rather than double counting it.
    j = np.searchsorted(starts, flat, side="right") - 1
    inside = (j >= 0) & (flat <= ends[np.clip(j, 0, n_l3 - 1)])

    counts = np.zeros((n_eps, n_l3), dtype="i8")
    np.add.at(counts, (owner[inside], j[inside]), 1)

    totals = valid.sum(axis=1)
    n_best = counts.max(axis=1)
    assigned = np.where(n_best > 0, counts.argmax(axis=1), -1)
    fraction = np.where(totals > 0, n_best / np.maximum(totals, 1), 0.0)

    if not (assigned >= 0).any():
        raise ValueError(
            f"No microstructure bin times fall inside any L3 profile window. The "
            f"microstructure spans {flat.min()} to {flat.max()} and the L3 file "
            f"spans {starts.min()} to {ends.max()}; the eps-bin file probably "
            f"belongs to a different glider or deployment."
        )

    _resolve_duplicates(assigned, fraction, n_l3)
    _report_assignment(ds, assigned, fraction)
    return assigned, fraction


def _resolve_duplicates(assigned: NDArray, fraction: NDArray, n_l3: int) -> None:
    """Keep the best claim where two microstructure profiles want one L3 profile."""
    claimed = assigned[assigned >= 0]
    for target in np.flatnonzero(np.bincount(claimed, minlength=n_l3) > 1):
        claimants = np.flatnonzero(assigned == target)
        keep = claimants[np.argmax(fraction[claimants])]
        assigned[claimants[claimants != keep]] = -1
        _log.warning(
            "L3 profile index %d is claimed by microstructure profiles %s; keeping "
            "%d (overlap %.2f) and discarding the rest.",
            target,
            claimants.tolist(),
            keep,
            fraction[keep],
        )


def _report_assignment(ds: xr.Dataset, assigned: NDArray, fraction: NDArray) -> None:
    """Log how well the microstructure profiles lined up with the L3 profiles."""
    ok = assigned >= 0
    _log.info(
        "Assigned %d/%d microstructure profiles to L3 profiles (median overlap "
        "%.3f, minimum %.3f).",
        ok.sum(),
        assigned.size,
        np.median(fraction[ok]) if ok.any() else np.nan,
        fraction[ok].min() if ok.any() else np.nan,
    )
    if ok.sum() < assigned.size / 2:
        _log.warning(
            "Only %d of %d microstructure profiles overlap an L3 profile window; "
            "check that both files are from the same deployment.",
            ok.sum(),
            assigned.size,
        )
    for i in np.flatnonzero(~ok):
        _log.warning(
            "Microstructure profile %d does not overlap any L3 profile window; "
            "not merged.",
            i,
        )
    for i in np.flatnonzero(ok & (fraction < 0.5)):
        _log.warning(
            "Microstructure profile %d has only %.0f%% of its bin times inside L3 "
            "profile index %d.",
            i,
            100 * fraction[i],
            assigned[i],
        )

    if "state" in ds:
        state = np.asarray(ds["state"].values)
        climbs = state == 2
        covered = np.zeros(state.size, dtype=bool)
        covered[assigned[ok]] = True
        uncovered = int((climbs & ~covered).sum())
        if uncovered:
            _log.info(
                "%d of %d climb profiles have no microstructure coverage.",
                uncovered,
                int(climbs.sum()),
            )


# Public functions


def parse_l3(file: str) -> tuple[xr.Dataset, float]:
    ds = xr.open_dataset(file, decode_timedelta=True).load()
    bin_size = _infer_bin_size(ds)
    ds.close()  # Will enable overwrite of existing l3 file.
    return ds, bin_size


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
            config["merged_variables"][v]["CF"],
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
        # Filter to valid depths within the bin range (also drops NaN).
        in_range = (eds_.depth >= depth_bins[0]) & (eds_.depth <= depth_bins[-1])
        eds_ = eds_.sel(time=in_range)
        if eds_.time.size < 1:
            _log.debug("No epsilon data")
            continue
        binned = eds_.groupby_bins("depth", depth_bins).mean()

        # groupby_bins drops bins that no sample fell in, so the result is only
        # as long as the bins this profile reached. Place the values by matching
        # bin centres rather than by position, which would misalign any profile
        # that did not span the whole depth grid.
        mids = np.array([interval.mid for interval in binned.depth_bins.values])
        keep, target = _depth_map(-np.asarray(ds.z.values, dtype="f8"), mids)

        for v in dissipation_variables:
            ds[v].values[target, i] = binned[v].values[keep]

    return ds


def _merge_attrs(ds_eps: xr.Dataset, name: str, config: dict) -> dict:
    """Attributes for an imported variable, the file winning over the config.

    pyturb writes its own CF metadata, so the config only fills gaps. The one
    exception is its `units: '-'` placeholder, which is dropped so the config can
    supply real units for files processed before pyturb started emitting them.
    """
    declared = (config.get("merged_variables") or {}).get(name) or {}
    attrs = dict(declared.get("CF") or {})

    from_file = {
        k: v
        for k, v in ds_eps[name].attrs.items()
        if k not in ("coordinates", "_FillValue")
    }
    if from_file.get("units") == "-":
        del from_file["units"]
    attrs.update(from_file)
    return attrs


def _select_eps_variables(
    ds: xr.Dataset, ds_eps: xr.Dataset, config: dict
) -> list[str]:
    """Variables to import: those declared in `merged_variables` and present.

    Declaring a name that L3 already holds is an error rather than an overwrite,
    which is what keeps the microstructure file's own temperature, salinity and
    thermodynamic variables out without needing a hard coded list of them.
    """
    declared = list(config.get("merged_variables") or {})
    present = [v for v in declared if v in ds_eps.data_vars]

    absent = [v for v in declared if v not in ds_eps.data_vars]
    if absent:
        # The q file variables are declared in the same section, so this is
        # ordinary rather than a problem.
        _log.debug("Declared variables absent from the microstructure file: %s", absent)

    if not present:
        raise ValueError(
            f"None of the merged_variables in the configuration ({declared}) are "
            f"present in the microstructure file. Declare the variables you want, "
            f"for example 'eps' and 'chi', under merged_variables."
        )

    for v in present:
        if v in ds.variables:
            raise ValueError(
                f"Merged variable '{v}' already exists in the L3 dataset. glide "
                f"will not overwrite its own fields; remove '{v}' from "
                f"merged_variables in your configuration. The microstructure "
                f"file's temperature, salinity, z, lat, lon and thermodynamic "
                f"variables duplicate quantities glide calculates itself."
            )
    return present


def merge_eps_bin(
    ds: xr.Dataset, ds_eps: xr.Dataset, bin_size: float, config: dict
) -> xr.Dataset:
    """Place depth-binned microstructure onto the L3 profile and depth grid.

    pyturb has already averaged the microstructure into depth bins, so nothing is
    re-binned here: the values are placed on the matching L3 bins, which requires
    the two grids to share a bin size. Each microstructure profile is assigned to
    the L3 profile it overlaps most in time (see `_assign_eps_profiles`), and the
    variables to import are those declared under `merged_variables` in the config.

    Parameters
    ----------
    ds : xr.Dataset
        L3 dataset on the (z, profile_id) grid, from `process_l2.bin_l2`.
    ds_eps : xr.Dataset
        Microstructure dataset from `ancillery.parse_eps_bin`.
    bin_size : float
        L3 depth bin size in metres.
    config : dict
        Configuration; imported variables are read from `merged_variables`.

    Returns
    -------
    xr.Dataset
        ``ds`` with the microstructure variables and their QC companions added,
        and the source file and instrument serial recorded in its attributes.
    """
    eps_bin_size = _eps_bin_size(ds_eps)
    _check_bin_size(bin_size, eps_bin_size)
    _log.info(
        "L3 and microstructure bins are both %g m; placing binned values directly",
        bin_size,
    )

    l3_centres = -np.asarray(ds["z"].values, dtype="f8")
    if not np.allclose(np.diff(l3_centres), bin_size, rtol=0, atol=_TOL):
        _log.warning(
            "L3 depth grid is not a contiguous run of %g m bins; microstructure "
            "bins with no matching L3 bin will be dropped.",
            bin_size,
        )
    eps_idx, l3_idx = _depth_map(
        l3_centres, np.asarray(ds_eps["depth"].values, dtype="f8")
    )

    variables = _select_eps_variables(ds, ds_eps, config)
    assigned, _ = _assign_eps_profiles(ds, ds_eps)

    dims = ("z", "profile_id")
    missing_dims = [d for d in dims if d not in ds.sizes]
    if missing_dims:
        raise ValueError(
            f"Expected an L3 dataset with dimensions {dims}, missing {missing_dims}."
        )
    shape = (ds.sizes["z"], ds.sizes["profile_id"])

    for name in variables:
        for v in (name, f"{name}_qc"):
            if v not in ds_eps.data_vars:
                continue
            is_flag = v.endswith("_qc")
            fill = qc.MISSING if is_flag else np.nan
            out = np.full(shape, fill, dtype="i1" if is_flag else "f4")

            source = ds_eps[v].values
            for eps_profile, target in enumerate(assigned):
                if target < 0:
                    continue
                out[l3_idx, target] = source[eps_profile, eps_idx]

            attrs = (
                dict(ds_eps[v].attrs) if is_flag else _merge_attrs(ds_eps, v, config)
            )
            ds[v] = (dims, out, attrs)

        if f"{name}_qc" in ds_eps.data_vars:
            ds[name].attrs["ancillary_variables"] = f"{name}_qc"

    ds.attrs.update(_provenance(ds_eps))
    return ds


def _provenance(ds_eps: xr.Dataset) -> dict:
    """Global attributes naming the microstructure file and the instrument.

    Deliberately minimal: everything else about the processing is recorded in the
    eps-bin file itself, which these two attributes are enough to find. The serial
    number comes from the per-profile variable rather than the global attribute,
    because the globals in an eps-bin file describe only the first p file it was
    built from.
    """
    attrs = {}

    source = ds_eps.encoding.get("source")
    if source is not None:
        attrs["microstructure_source_file"] = str(source)

    if "instrument_sn" in ds_eps.variables:
        serial = np.unique(ds_eps["instrument_sn"].values.astype(str))
        attrs["microstructure_instrument_sn"] = ", ".join(serial)
        if serial.size > 1:
            _log.warning(
                "Microstructure file contains several instrument serial numbers: %s",
                serial,
            )
    elif "instrument_sn" in ds_eps.attrs:
        attrs["microstructure_instrument_sn"] = str(ds_eps.attrs["instrument_sn"])

    return attrs
