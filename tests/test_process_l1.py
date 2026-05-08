from importlib import resources

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import glide.gliderdac as gd
import glide.process_l1 as pl1
import glide.profiles as prof
import glide.qc as qc
from glide.config import load_config


def get_test_data(sn: str = "684", ftype: str = "sbd") -> xr.Dataset:
    return (
        pd.read_csv(str(resources.files("tests").joinpath(f"data/osu{sn}.{ftype}.csv")))
        .set_index("i")
        .to_xarray()
    )


def get_realtime_test_data(segment: int, ftype: str = "sbd") -> xr.Dataset:
    """Load real-time test data for osu685 segments 27-30."""
    return (
        pd.read_csv(
            str(
                resources.files("tests").joinpath(
                    f"data/osu685-2025-056-0-{segment}.{ftype}.csv"
                )
            )
        )
        .set_index("i")
        .to_xarray()
    )


def test_pair_input_files_single(tmp_path) -> None:
    flt = tmp_path / "g-2025-001.sbd.csv"
    sci = tmp_path / "g-2025-001.tbd.csv"
    flt.write_text("")
    sci.write_text("")

    pairs = pl1.pair_input_files(str(flt), str(sci))
    assert pairs == [(str(flt), str(sci))]


def test_pair_input_files_glob_pairs_by_stem(tmp_path) -> None:
    stems = ["g-2025-001", "g-2025-002", "g-2025-003"]
    for s in stems:
        (tmp_path / f"{s}.sbd.csv").write_text("")
        (tmp_path / f"{s}.tbd.csv").write_text("")

    pairs = pl1.pair_input_files(
        str(tmp_path / "*.sbd.csv"), str(tmp_path / "*.tbd.csv")
    )
    assert len(pairs) == 3
    for flt, sci in pairs:
        assert flt.endswith(".sbd.csv")
        assert sci.endswith(".tbd.csv")
        assert flt.replace(".sbd.csv", "") == sci.replace(".tbd.csv", "")


def test_pair_input_files_unpaired_raises(tmp_path) -> None:
    (tmp_path / "g-001.sbd.csv").write_text("")
    (tmp_path / "g-002.sbd.csv").write_text("")
    (tmp_path / "g-001.tbd.csv").write_text("")  # g-002.tbd.csv is missing

    import pytest

    with pytest.raises(ValueError, match="Unpaired"):
        pl1.pair_input_files(str(tmp_path / "*.sbd.csv"), str(tmp_path / "*.tbd.csv"))


def test_pair_input_files_skip_unpaired(tmp_path, caplog) -> None:
    import logging

    (tmp_path / "g-001.sbd.csv").write_text("")
    (tmp_path / "g-002.sbd.csv").write_text("")
    (tmp_path / "g-001.tbd.csv").write_text("")

    with caplog.at_level(logging.WARNING, logger="glide.process_l1"):
        pairs = pl1.pair_input_files(
            str(tmp_path / "*.sbd.csv"),
            str(tmp_path / "*.tbd.csv"),
            skip_unpaired=True,
        )

    assert len(pairs) == 1
    assert pairs[0][0].endswith("g-001.sbd.csv")
    assert any("Unpaired" in rec.message for rec in caplog.records)


def test_pair_input_files_wrong_kind_raises(tmp_path) -> None:
    """A .tbd file in the flight pattern should error — strict by design."""
    (tmp_path / "g-001.tbd.csv").write_text("")
    (tmp_path / "g-001.sbd.csv").write_text("")

    import pytest

    with pytest.raises(ValueError, match="not a flight file"):
        pl1.pair_input_files(str(tmp_path / "*.tbd.csv"), str(tmp_path / "*.sbd.csv"))


def test_pair_input_files_unrecognized_name(tmp_path) -> None:
    bogus = tmp_path / "not_a_glider_file.txt"
    bogus.write_text("")

    import pytest

    with pytest.raises(ValueError, match="does not look like"):
        pl1.pair_input_files(str(bogus), str(bogus))


def test_pair_input_files_no_match(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="No flight files matched"):
        pl1.pair_input_files(
            str(tmp_path / "missing*.sbd.csv"), str(tmp_path / "*.tbd.csv")
        )


def test_format_variables() -> None:
    config = load_config()
    sbd = pl1._format_variables(get_test_data("684", "sbd"), config)
    assert hasattr(sbd, "time")
    assert hasattr(sbd.time, "units")
    tbd = pl1._format_variables(get_test_data("684", "tbd"), config)
    assert hasattr(tbd, "time")
    assert hasattr(tbd.time, "units")


def test_parse_l1() -> None:
    pl1.parse_l1(get_test_data("684", "sbd"))
    pl1.parse_l1(get_test_data("684", "tbd"))


def test_merge_creates_qc_for_flight_variables() -> None:
    """Variables from flight data with track_qc: True must have QC counterparts
    in the merged dataset. lat and lon only exist in flight data; their _qc
    variables are dropped during merge interpolation, so merge() must re-create
    them from scratch."""
    config = load_config()

    n_flt = 20
    flt_time = np.arange(n_flt, dtype="f8") * 10.0  # 0, 10, ..., 190 s
    lat_vals = np.linspace(44.0, 45.0, n_flt)
    lon_vals = np.linspace(-125.0, -124.0, n_flt)
    lat_vals[3:5] = np.nan  # Simulate missing dead-reckoning positions

    flt = xr.Dataset(
        {
            "lat": (
                "time",
                lat_vals,
                {
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                    "valid_min": -90.0,
                    "valid_max": 90.0,
                },
            ),
            "lon": (
                "time",
                lon_vals,
                {
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                    "valid_min": -180.0,
                    "valid_max": 180.0,
                },
            ),
        },
        coords={"time": flt_time},
    )
    flt = qc.init_qc(flt, ["lat", "lon"], config=config)

    n_sci = 15
    sci_time = np.linspace(5.0, 185.0, n_sci)
    pressure_vals = np.linspace(0.0, 100.0, n_sci)

    sci = xr.Dataset(
        {
            "pressure": (
                "time",
                pressure_vals,
                {
                    "long_name": "Pressure",
                    "standard_name": "sea_water_pressure",
                    "valid_min": 0.0,
                    "valid_max": 2000.0,
                },
            ),
        },
        coords={"time": sci_time},
    )
    sci = qc.init_qc(sci, ["pressure"], config=config)

    merged = pl1.merge(flt, sci, config, "science")

    # Interpolated flight variables must be present
    assert "lat" in merged
    assert "lon" in merged

    # QC variables must exist for all track_qc: True variables in the merged output
    assert "lat_qc" in merged, "lat_qc missing after merge"
    assert "lon_qc" in merged, "lon_qc missing after merge"

    # QC arrays must align with the science time dimension
    assert merged.lat_qc.shape == (n_sci,)
    assert merged.lon_qc.shape == (n_sci,)

    # Non-NaN values must be flagged as interpolated (8); NaN as missing (9)
    lat_merged = merged.lat.values
    lat_qc_vals = merged.lat_qc.values
    assert np.all(lat_qc_vals[np.isfinite(lat_merged)] == 8)
    assert np.all(lat_qc_vals[~np.isfinite(lat_merged)] == 9)

    # ancillary_variables attribute must point to the QC variable
    assert merged.lat.attrs.get("ancillary_variables") == "lat_qc"
    assert merged.lon.attrs.get("ancillary_variables") == "lon_qc"

    # Science QC variables must be preserved through the merge
    assert "pressure_qc" in merged, "pressure_qc missing after merge"


def test_merge_does_not_overwrite_existing_qc() -> None:
    """If a variable's QC counterpart already exists in the base (science) dataset,
    merge() must not overwrite it."""
    config = load_config()

    n_flt = 10
    flt_time = np.arange(n_flt, dtype="f8") * 20.0

    flt = xr.Dataset(
        {
            "lat": (
                "time",
                np.linspace(44.0, 45.0, n_flt),
                {"long_name": "Latitude", "standard_name": "latitude"},
            ),
        },
        coords={"time": flt_time},
    )
    flt = qc.init_qc(flt, ["lat"], config=config)

    n_sci = 8
    sci_time = np.linspace(10.0, 170.0, n_sci)
    sentinel_flags = np.full(n_sci, 3, dtype="b")  # Intentionally distinct flag value

    sci = xr.Dataset(
        {
            "lat": (
                "time",
                np.linspace(44.1, 44.9, n_sci),
                {"long_name": "Latitude", "standard_name": "latitude"},
            ),
            "lat_qc": (
                "time",
                sentinel_flags,
                {"flag_meanings": "sentinel", "flag_values": np.array([3], dtype="b")},
            ),
        },
        coords={"time": sci_time},
    )

    merged = pl1.merge(flt, sci, config, "science")

    assert "lat_qc" in merged
    assert np.all(merged.lat_qc.values == 3), (
        "Pre-existing lat_qc must not be overwritten"
    )


def _make_surface_state_inputs(
    n: int = 100,
    dive_slice: slice = slice(20, 40),
    climb_slice: slice = slice(40, 60),
    gps_fix_indices: list[int] | None = None,
    pressure: np.ndarray | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build synthetic (ds, flt) pair for assign_surface_state tests."""
    if gps_fix_indices is None:
        gps_fix_indices = [5, 10, 70, 80]

    time = np.arange(n, dtype=float)
    state = np.full(n, -1, dtype="b")
    state[dive_slice] = 1
    state[climb_slice] = 2

    ds_vars: dict = {
        "state": (
            "time",
            state,
            dict(
                long_name="Glider state",
                flag_values=np.array([-1, 0, 1, 2], "b"),
                flag_meanings="unknown surface dive climb",
            ),
        ),
    }
    if pressure is not None:
        ds_vars["pressure"] = ("time", pressure)
    ds = xr.Dataset(ds_vars, coords={"time": time})

    gps_lat = np.full(n, np.nan)
    for idx in gps_fix_indices:
        gps_lat[idx] = 45.0
    flt = xr.Dataset(
        {"m_gps_lat": ("time", gps_lat), "m_present_time": ("time", time)},
        coords={"time": time},
    )
    return ds, flt


def test_assign_surface_state() -> None:
    """Shallow unknown points near GPS fixes are marked surface; deep ones are not."""
    n = 100
    pressure = np.full(n, np.nan)
    # Surface regions: indices 0-19 and 60-99 → shallow
    pressure[:20] = 0.5
    pressure[60:] = 0.5
    # Dive/climb region: deep
    pressure[20:60] = 50.0

    ds, flt = _make_surface_state_inputs(n=n, pressure=pressure)

    result = prof.assign_surface_state(ds, flt, dt=15.0, surface_pressure=2.0)

    # Shallow points at GPS fix times → surface
    assert result.state.values[5] == 0, "Shallow point at GPS fix should be surface"
    assert result.state.values[10] == 0, "Shallow point at GPS fix should be surface"
    assert result.state.values[70] == 0, "Shallow point at GPS fix should be surface"
    assert result.state.values[80] == 0, "Shallow point at GPS fix should be surface"

    # Dive/climb states are unchanged
    assert np.all(result.state.values[20:40] == 1), "Dive states should be unchanged"
    assert np.all(result.state.values[40:60] == 2), "Climb states should be unchanged"


def test_assign_surface_state_pressure_gate() -> None:
    """Deep unknown points near GPS fixes must NOT be assigned surface state.

    This is the regression test for the bug where the flight computer's high
    GPS sampling rate (many fixes per surfacing) caused the time-proximity
    window to reach into adjacent dives at depth.
    """
    n = 100
    pressure = np.full(n, 50.0)  # all points at 50 dbar — definitely not surface
    pressure[5] = 0.5  # only index 5 is actually shallow

    # GPS fix at index 5 (shallow, outside dive/climb) and index 65 (deep,
    # outside dive/climb so state starts as -1 — the exact case where the bug
    # would erroneously assign state 0).
    ds, flt = _make_surface_state_inputs(
        n=n, gps_fix_indices=[5, 65], pressure=pressure
    )

    result = prof.assign_surface_state(ds, flt, dt=15.0, surface_pressure=2.0)

    # Only the shallow point near a GPS fix should become surface
    assert result.state.values[5] == 0, "Shallow point at GPS fix should be surface"

    # Deep point coinciding with a GPS fix must stay unknown
    assert result.state.values[65] == -1, (
        "Deep point at GPS fix must not be marked surface"
    )

    # Deep points near but not at the GPS fix must also stay unknown
    for idx in [61, 63, 67, 69]:
        assert result.state.values[idx] == -1, (
            f"Deep point at index {idx} must not be marked surface"
        )


def test_assign_surface_state_no_flight_data() -> None:
    """Without flight data the GPS-proximity check is skipped, but shallow
    unknown segments are still classified as surface by the pressure-only pass."""
    n = 20
    state = np.full(n, -1, dtype="b")
    state[5:15] = 1
    pressure = np.full(n, 0.5)
    pressure[5:15] = 50.0  # dive at depth

    ds = xr.Dataset(
        {"state": ("time", state), "pressure": ("time", pressure)},
        coords={"time": np.arange(n, dtype=float)},
    )

    result = prof.assign_surface_state(ds, flt=None, surface_pressure=2.0)

    assert np.all(result.state.values[5:15] == 1), "Dive states unchanged"
    assert np.all(result.state.values[:5] == 0), "Pre-dive shallow → surface"
    assert np.all(result.state.values[15:] == 0), "Post-dive shallow → surface"


def test_assign_surface_state_pre_first_dive_and_post_last_climb() -> None:
    """Time before the first dive and after the last climb is classified as
    surface when pressure stays below surface_pressure."""
    n = 50
    state = np.full(n, -1, dtype="b")
    state[10:20] = 1  # dive
    state[20:30] = 2  # climb
    pressure = np.full(n, 0.3)
    pressure[10:20] = np.linspace(0.3, 80, 10)
    pressure[20:30] = np.linspace(80, 0.3, 10)

    ds = xr.Dataset(
        {"state": ("time", state), "pressure": ("time", pressure)},
        coords={"time": np.arange(n, dtype=float)},
    )

    result = prof.assign_surface_state(ds, flt=None, surface_pressure=2.0)

    assert np.all(result.state.values[:10] == 0), "Pre-first-dive → surface"
    assert np.all(result.state.values[30:] == 0), "Post-last-climb → surface"
    assert np.all(result.state.values[10:20] == 1)
    assert np.all(result.state.values[20:30] == 2)


def test_assign_surface_state_all_nan_pressure_is_surface() -> None:
    """Surface segments where the science pressure sensor is out of water
    (all NaN) must still be classified as surface."""
    n = 30
    state = np.full(n, -1, dtype="b")
    state[5:15] = 1  # dive
    state[15:20] = 2  # climb
    pressure = np.full(n, np.nan)
    pressure[5:15] = np.linspace(2, 80, 10)
    pressure[15:20] = np.linspace(80, 2, 5)
    # Indices 0–4 and 20–29 have NaN pressure (surface, sensor out of water)

    ds = xr.Dataset(
        {"state": ("time", state), "pressure": ("time", pressure)},
        coords={"time": np.arange(n, dtype=float)},
    )

    result = prof.assign_surface_state(ds, flt=None, surface_pressure=2.0)

    assert np.all(result.state.values[:5] == 0), "Pre-dive all-NaN → surface"
    assert np.all(result.state.values[20:] == 0), "Post-climb all-NaN → surface"
    assert np.all(result.state.values[5:15] == 1)
    assert np.all(result.state.values[15:20] == 2)


def test_assign_surface_state_protects_apex_nan_gap() -> None:
    """An all-NaN unknown gap sandwiched between a dive and a climb is the
    underwater apex (the apex absorber couldn't split it because pressure
    was missing).  It must NOT be reclassified as surface."""
    n = 30
    state = np.full(n, -1, dtype="b")
    state[5:13] = 1  # dive
    state[16:25] = 2  # climb
    pressure = np.full(n, np.nan)
    pressure[5:13] = np.linspace(2, 80, 8)
    pressure[16:25] = np.linspace(80, 2, 9)
    # Indices 13–15 NaN (sensor briefly missing at the apex)

    ds = xr.Dataset(
        {"state": ("time", state), "pressure": ("time", pressure)},
        coords={"time": np.arange(n, dtype=float)},
    )

    result = prof.assign_surface_state(ds, flt=None, surface_pressure=2.0)

    assert np.all(result.state.values[13:16] == -1), (
        "Apex NaN gap must stay unknown, not flip to surface"
    )
    assert np.all(result.state.values[5:13] == 1)
    assert np.all(result.state.values[16:25] == 2)


def test_assign_surface_state_keeps_aborted_dive_unknown() -> None:
    """An unknown segment that contains a partial dive (pressure exceeds
    surface_pressure) must not be reclassified as surface."""
    n = 30
    state = np.full(n, -1, dtype="b")
    state[20:25] = 2  # climb
    pressure = np.full(n, 0.3)
    # Aborted dive between samples 5–15: pressure briefly reaches 4 dbar
    pressure[5:15] = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    pressure[20:25] = np.linspace(40, 0.3, 5)

    ds = xr.Dataset(
        {"state": ("time", state), "pressure": ("time", pressure)},
        coords={"time": np.arange(n, dtype=float)},
    )

    result = prof.assign_surface_state(ds, flt=None, surface_pressure=2.0)

    # The whole pre-climb segment contains a sample > surface_pressure → stays unknown
    assert np.all(result.state.values[:20] == -1)
    # Post-climb stays surface (pressure shallow throughout)
    assert np.all(result.state.values[25:] == 0)


def test_add_velocity_groups_by_velocity_reports() -> None:
    """Test that add_velocity groups profiles by velocity report times."""
    from glide.config import load_config

    config = load_config()

    # Create L2-like dataset with 2 dive/climb cycles
    # Times are in seconds, spaced to represent realistic dive durations
    n = 100
    time = np.arange(n, dtype=float) * 100  # 100s spacing
    pressure = np.zeros(n)
    # Cycle 1: dive 10-30, climb 30-50
    pressure[10:30] = np.linspace(0, 100, 20)
    pressure[30:50] = np.linspace(100, 0, 20)
    # Cycle 2: dive 60-80, climb 80-95
    pressure[60:80] = np.linspace(0, 100, 20)
    pressure[80:95] = np.linspace(100, 0, 15)

    dive_id = np.full(n, -1, dtype="i4")
    climb_id = np.full(n, -1, dtype="i4")
    dive_id[10:30] = 1
    climb_id[30:50] = 1
    dive_id[60:80] = 2
    climb_id[80:95] = 2

    ds = xr.Dataset(
        {
            "pressure": ("time", pressure),
            "dive_id": ("time", dive_id),
            "climb_id": ("time", climb_id),
            "lat": ("time", np.full(n, 45.0)),
            "lon": ("time", np.full(n, -125.0)),
        },
        coords={"time": time},
    )

    # Flight data with velocity reports at surfacing times
    # Times must be > 600s apart to be recognized as separate surfacings
    flt_time = np.array([500.0, 5500.0, 9800.0])
    flt = xr.Dataset(
        {
            "m_present_time": ("i", flt_time),
            "m_water_vx": ("i", [0.1, 0.2, 0.3]),
            "m_water_vy": ("i", [-0.05, -0.1, -0.15]),
        },
    )

    result = prof.add_velocity(ds, config, flt=flt)

    # Velocity at t=55 should capture cycle 1 (profiles 10-50)
    # Velocity at t=98 should capture cycle 2 (profiles 60-95)
    assert "u" in result
    assert result.sizes["time_uv"] >= 2
    assert np.isfinite(result.u.values).sum() >= 2


def test_realtime_velocity_processing() -> None:
    """Test that velocity variables are added when processing real-time file pairs.

    In real-time mode, we process each sbd/tbd pair individually.
    Individual files may not contain complete dive cycles, so velocity
    may be NaN. The key test is that velocity variables are always added.
    """
    from glide.config import load_config

    config = load_config()

    # Process segments 27-30 individually
    segments = [27, 28, 29, 30]
    results = []

    for seg in segments:
        sbd = get_realtime_test_data(seg, "sbd")
        tbd = get_realtime_test_data(seg, "tbd")

        # Format and parse
        flt_raw = sbd.copy()
        flt = pl1.format_l1(pl1.parse_l1(sbd), config)
        sci = pl1.format_l1(pl1.parse_l1(tbd), config)

        # Apply QC (catch GPS QC failures for small test files)
        try:
            flt = pl1.apply_qc(flt, config)
        except (ValueError, AttributeError):
            # GPS QC may fail on small test files - use basic QC instead
            from glide import qc

            flt = qc.init_qc(flt, config=config)
            flt = qc.apply_bounds(flt)
            flt = qc.time(flt)
            dim = list(flt.sizes.keys())[0]
            flt = flt.swap_dims({dim: "time"})
            flt = qc.interpolate_missing(flt, config)

        try:
            sci = pl1.apply_qc(sci, config)
        except (ValueError, AttributeError):
            from glide import qc

            sci = qc.init_qc(sci, config=config)
            sci = qc.apply_bounds(sci)
            sci = qc.time(sci)
            dim = list(sci.sizes.keys())[0]
            sci = sci.swap_dims({dim: "time"})
            sci = qc.interpolate_missing(sci, config)

        # Merge
        merged = pl1.merge(flt, sci, config, "science")
        merged = pl1.calculate_thermodynamics(merged, config)

        # Get profiles
        out = prof.get_profiles(merged, shallowest_profile=5.0, min_surface_time=180.0)

        # Assign surface state and add velocity
        out = prof.assign_surface_state(out, flt=flt_raw)
        out = prof.add_velocity(out, config, flt=flt_raw)

        results.append(
            {
                "segment": seg,
                "has_time_uv": "time_uv" in out,
                "has_u": "u" in out,
                "has_v": "v" in out,
                "n_cycles": out.sizes.get("time_uv", 0),
                "n_valid_u": int(np.isfinite(out.u.values).sum()) if "u" in out else 0,
                "ds": out,
            }
        )

    # All files should have velocity variables (even if NaN)
    for r in results:
        assert r["has_time_uv"], f"Segment {r['segment']} missing time_uv"
        assert r["has_u"], f"Segment {r['segment']} missing u"
        assert r["has_v"], f"Segment {r['segment']} missing v"


def test_realtime_velocity_merged() -> None:
    """Test velocity detection when merging all real-time segments together.

    When multiple segments are combined, we should have enough data to
    detect complete dive cycles and assign velocity.
    """
    from glide.config import load_config

    config = load_config()

    # Concatenate all segments
    segments = [27, 28, 29, 30]
    all_sbd = []
    all_tbd = []

    for seg in segments:
        all_sbd.append(get_realtime_test_data(seg, "sbd"))
        all_tbd.append(get_realtime_test_data(seg, "tbd"))

    # Concatenate along the index dimension
    sbd = xr.concat(all_sbd, dim="i")
    tbd = xr.concat(all_tbd, dim="i")

    # Re-index to have unique i values
    sbd = sbd.assign_coords(i=np.arange(sbd.sizes["i"]))
    tbd = tbd.assign_coords(i=np.arange(tbd.sizes["i"]))

    # Format and parse
    flt_raw = sbd.copy()
    flt = pl1.format_l1(pl1.parse_l1(sbd), config)
    sci = pl1.format_l1(pl1.parse_l1(tbd), config)

    # Apply QC
    try:
        flt = pl1.apply_qc(flt, config)
    except (ValueError, AttributeError):
        from glide import qc

        flt = qc.init_qc(flt, config=config)
        flt = qc.apply_bounds(flt)
        flt = qc.time(flt)
        dim = list(flt.sizes.keys())[0]
        flt = flt.swap_dims({dim: "time"})
        flt = qc.interpolate_missing(flt, config)

    try:
        sci = pl1.apply_qc(sci, config)
    except (ValueError, AttributeError):
        from glide import qc

        sci = qc.init_qc(sci, config=config)
        sci = qc.apply_bounds(sci)
        sci = qc.time(sci)
        dim = list(sci.sizes.keys())[0]
        sci = sci.swap_dims({dim: "time"})
        sci = qc.interpolate_missing(sci, config)

    # Merge
    merged = pl1.merge(flt, sci, config, "science")
    merged = pl1.calculate_thermodynamics(merged, config)

    # Get profiles
    out = prof.get_profiles(merged, shallowest_profile=5.0, min_surface_time=180.0)

    # Check that we detected some profiles
    n_dives = (out.dive_id.values >= 0).sum()

    # Assign surface state and add velocity
    out = prof.assign_surface_state(out, flt=flt_raw)
    out = prof.add_velocity(out, config, flt=flt_raw)

    # Should have velocity variables
    assert "time_uv" in out, "Missing time_uv"
    assert "u" in out, "Missing u"
    assert "v" in out, "Missing v"

    # If we detected profiles, we should have some velocity estimates
    # (may still be NaN if velocity data not in these segments)
    if n_dives > 0:
        assert out.sizes["time_uv"] >= 1, "Expected at least one velocity entry"


def test_add_gps_fixes_creates_gps_dimension() -> None:
    """Only valid (non-NaN) GPS fixes should appear on the time_gps dimension,
    preserving their original timestamps."""
    config = load_config()

    n = 10
    time_vals = np.arange(n, dtype="f8") * 100.0  # 0, 100, ..., 900 s
    lat_vals = np.full(n, np.nan)
    lon_vals = np.full(n, np.nan)
    # Three valid fixes embedded among NaNs
    fix_indices = [2, 5, 8]
    lat_vals[fix_indices] = [44.0, 44.5, 45.0]
    lon_vals[fix_indices] = [-125.0, -124.5, -124.0]

    flt = xr.Dataset(
        {"lat_gps": ("time", lat_vals), "lon_gps": ("time", lon_vals)},
        coords={"time": time_vals},
    )
    ds = xr.Dataset(coords={"time": time_vals})

    result = prof.add_gps_fixes(ds, flt, config)

    assert "time_gps" in result.dims, "time_gps dimension missing"
    assert "lat_gps" in result, "lat_gps missing"
    assert "lon_gps" in result, "lon_gps missing"

    assert result.sizes["time_gps"] == 3
    assert np.all(np.isfinite(result.lat_gps.values)), "lat_gps must not contain NaN"
    assert np.all(np.isfinite(result.lon_gps.values)), "lon_gps must not contain NaN"
    assert np.allclose(result.lat_gps.values, [44.0, 44.5, 45.0])
    assert np.allclose(result.time_gps.values, [200.0, 500.0, 800.0])


def test_add_gps_fixes_no_gps_variables() -> None:
    """Dataset must be returned unchanged when the flight data has no GPS variables."""
    config = load_config()

    flt = xr.Dataset(coords={"time": np.arange(5, dtype="f8")})
    ds = xr.Dataset(coords={"time": np.arange(5, dtype="f8")})

    result = prof.add_gps_fixes(ds, flt, config)

    assert "time_gps" not in result.dims
    assert "lat_gps" not in result
    assert "lon_gps" not in result


def test_add_gps_fixes_all_nan() -> None:
    """Dataset must be returned unchanged when all GPS fixes are NaN."""
    config = load_config()

    n = 5
    flt = xr.Dataset(
        {
            "lat_gps": ("time", np.full(n, np.nan)),
            "lon_gps": ("time", np.full(n, np.nan)),
        },
        coords={"time": np.arange(n, dtype="f8")},
    )
    ds = xr.Dataset(coords={"time": np.arange(n, dtype="f8")})

    result = prof.add_gps_fixes(ds, flt, config)

    assert "time_gps" not in result.dims
    assert "lat_gps" not in result
    assert "lon_gps" not in result


def test_get_profiles_assigns_profile_id() -> None:
    """profile_id increments by 1 for each descent and each ascent in time order."""
    # Two triangular dives: 0..max..0..max..0. A leading NaN sidesteps a
    # profinder edge case where valid_idx is undefined for NaN-free input.
    pressure = np.concatenate(
        [
            np.array([np.nan]),
            np.linspace(1, 50, 50),
            np.linspace(50, 1, 50),
            np.linspace(1, 50, 50),
            np.linspace(50, 1, 50),
        ]
    )
    time = np.arange(len(pressure), dtype="f8")
    ds = xr.Dataset(
        {"pressure": ("time", pressure)},
        coords={"time": time},
    )

    result = prof.get_profiles(ds, shallowest_profile=5.0, min_surface_time=60.0)

    assert "profile_id" in result
    pid = result.profile_id.values
    unique = sorted({int(p) for p in pid if p >= 0})
    # At least 4 profiles (2 dives + 2 climbs), id'd 1..N sequentially.
    assert len(unique) >= 4
    assert unique == list(range(1, len(unique) + 1))

    # profile_id matches the dive_id/climb_id structure: dive_id == k gets
    # profile_id 2k-1, the immediately following climb_id == k gets 2k.
    dive_ids = sorted({int(d) for d in result.dive_id.values if d >= 0})
    climb_ids = sorted({int(c) for c in result.climb_id.values if c >= 0})
    for k in dive_ids:
        dive_pids = np.unique(pid[result.dive_id.values == k])
        assert len(dive_pids) == 1
        assert dive_pids[0] == 2 * k - 1
    for k in climb_ids:
        climb_pids = np.unique(pid[result.climb_id.values == k])
        assert len(climb_pids) == 1
        assert climb_pids[0] == 2 * k


def _make_synthetic_l2_ds(
    *,
    profile_id: np.ndarray,
    state: np.ndarray,
    u: float = 0.12,
    v: float = -0.05,
    t_uv: float = 20.0,
) -> xr.Dataset:
    """Build a minimal L2-shaped dataset for emit_ioos_profiles tests."""
    time = np.arange(len(profile_id), dtype="f8") * 10.0
    n_temp = np.where(profile_id >= 0, np.linspace(12.0, 13.0, len(profile_id)), np.nan)
    return xr.Dataset(
        {
            "temperature": ("time", n_temp),
            "profile_id": ("time", profile_id),
            "state": ("time", state),
            "dive_id": ("time", np.where(state == 1, 1, -1).astype("i4")),
            "climb_id": ("time", np.where(state == 2, 1, -1).astype("i4")),
            "u": ("time_uv", np.array([u])),
            "v": ("time_uv", np.array([v])),
            "time_uv": ("time_uv", np.array([t_uv])),
            "lat_uv": ("time_uv", np.array([45.0])),
            "lon_uv": ("time_uv", np.array([-123.0])),
        },
        coords={"time": time},
    )


def test_emit_ioos_profiles_writes_scalar_velocity(tmp_path) -> None:
    """A profile with finite velocity is emitted as an NGDAC-shaped scalar file."""
    config = load_config()
    ds = _make_synthetic_l2_ds(
        profile_id=np.array([-1, 1, 1, 1, -1, -1], dtype="i4"),
        state=np.array([0, 1, 1, 2, 0, 0], dtype="i1"),
    )

    written = gd.emit_ioos_profiles(ds, tmp_path, "test_glider", ngdac=config["ngdac"])

    assert len(written) == 1
    out = xr.open_dataset(written[0])
    try:
        # NGDAC contract: scalar u, v, time_uv, lat_uv, lon_uv, profile_id
        for v in ("u", "v", "time_uv", "lat_uv", "lon_uv", "profile_id"):
            assert v in out
            assert out[v].ndim == 0, f"{v} must be scalar"
        assert float(out.u.values) == 0.12
        assert float(out.v.values) == -0.05
        assert int(out.profile_id.values) == 1
        # Excluded variables / dims (segment_id no longer emitted)
        for v in ("dive_id", "climb_id", "state", "segment_id"):
            assert v not in out
        assert "time_uv" not in out.dims
        # Only the in-profile time points are kept
        assert out.sizes["time"] == 3
    finally:
        out.close()


def test_emit_ioos_profiles_skips_when_velocity_nan(tmp_path) -> None:
    """A profile in a segment with NaN u/v is skipped (no file written)."""
    config = load_config()
    ds = _make_synthetic_l2_ds(
        profile_id=np.array([-1, 1, 1, -1], dtype="i4"),
        state=np.array([0, 1, 2, 0], dtype="i1"),
        u=float("nan"),
        v=float("nan"),
        t_uv=15.0,
    )

    written = gd.emit_ioos_profiles(ds, tmp_path, "test_glider", ngdac=config["ngdac"])

    assert written == []
    assert list(tmp_path.glob("*.nc")) == []


def test_emit_ioos_profiles_idempotent(tmp_path) -> None:
    """Re-running with the same data does not write new files."""
    config = load_config()
    ds = _make_synthetic_l2_ds(
        profile_id=np.array([-1, 1, 1, -1], dtype="i4"),
        state=np.array([0, 1, 2, 0], dtype="i1"),
        u=0.1,
        v=0.0,
        t_uv=15.0,
    )

    first = gd.emit_ioos_profiles(ds, tmp_path, "test_glider", ngdac=config["ngdac"])
    assert len(first) == 1

    second = gd.emit_ioos_profiles(ds, tmp_path, "test_glider", ngdac=config["ngdac"])
    assert second == []
    assert len(list(tmp_path.glob("*.nc"))) == 1

    forced = gd.emit_ioos_profiles(
        ds, tmp_path, "test_glider", force=True, ngdac=config["ngdac"]
    )
    assert len(forced) == 1
    assert len(list(tmp_path.glob("*.nc"))) == 1


def test_emit_ioos_profiles_writes_ngdac_structural_vars(tmp_path) -> None:
    """platform, crs, profile_time, profile_lat, profile_lon, and configured
    instrument scalars are all written into each emitted file."""
    config = load_config()
    ds = _make_synthetic_l2_ds(
        profile_id=np.array([-1, 1, 1, 1, -1, -1], dtype="i4"),
        state=np.array([0, 1, 1, 2, 0, 0], dtype="i1"),
    )
    # Add the lat/lon arrays that profile_lat/lon are derived from.
    ds["lat"] = ("time", np.linspace(45.0, 45.5, ds.sizes["time"]))
    ds["lon"] = ("time", np.linspace(-123.0, -122.5, ds.sizes["time"]))

    instruments = {
        "instrument_ctd": {
            "long_name": "Test CTD",
            "make_model": "Sea-Bird GPCTD",
            "serial_number": "9264",
            "type": "instrument",
            "platform": "platform",
        },
        "instrument_oxygen": {
            "long_name": "Test Oxygen Sensor",
            "make_model": "Aanderaa Optode 4831",
            "serial_number": "416",
            "type": "instrument",
            "platform": "platform",
        },
    }

    written = gd.emit_ioos_profiles(
        ds,
        tmp_path,
        "synthglider",
        instruments=instruments,
        ngdac=config["ngdac"],
    )
    assert len(written) == 1
    out = xr.open_dataset(written[0])
    try:
        # platform and crs scalars present.
        assert "platform" in out and out.platform.ndim == 0
        assert "crs" in out and out.crs.ndim == 0
        assert out.platform.attrs.get("id") == "synthglider"
        assert out.platform.attrs.get("type") == "platform"
        # platform.instrument enumerates the configured instruments.
        assert "instrument_ctd" in out.platform.attrs.get("instrument", "")
        assert "instrument_oxygen" in out.platform.attrs.get("instrument", "")
        # CRS is WGS84 boilerplate.
        assert out.crs.attrs.get("epsg_code") == "EPSG:4326"

        # Configured instruments emitted as scalars with their attrs.
        for name, attrs in instruments.items():
            assert name in out and out[name].ndim == 0
            for k, v in attrs.items():
                assert out[name].attrs.get(k) == v, (
                    f"{name}.{k}: expected {v}, got {out[name].attrs.get(k)}"
                )

        # Profile-center variables.
        for v in ("profile_time", "profile_lat", "profile_lon"):
            assert v in out and out[v].ndim == 0
        assert np.isfinite(out.profile_lat.values)
        assert np.isfinite(out.profile_lon.values)
        # Profile center is the mean of the in-profile lat/lon (indices 1,2,3).
        expected_lat = float(np.mean(ds.lat.values[1:4]))
        assert abs(float(out.profile_lat.values) - expected_lat) < 1e-9
    finally:
        out.close()


def test_emit_ioos_profiles_yo_yo_shares_time_uv(tmp_path) -> None:
    """Profiles in the same segment share time_uv (canonical NGDAC grouping)."""
    config = load_config()
    # surface, dive, climb, dive, climb, surface — 4 profiles in one segment
    ds = _make_synthetic_l2_ds(
        profile_id=np.array([-1, 1, 2, 3, 4, -1], dtype="i4"),
        state=np.array([0, 1, 2, 1, 2, 0], dtype="i1"),
        u=0.2,
        v=0.1,
        t_uv=25.0,
    )

    written = gd.emit_ioos_profiles(ds, tmp_path, "test_glider", ngdac=config["ngdac"])
    assert len(written) == 4

    time_uvs = []
    for f in written:
        out = xr.open_dataset(f)
        time_uvs.append(float(out.time_uv.values))
        out.close()

    # All four profiles share the same scalar time_uv (= the segment's value).
    assert all(t == time_uvs[0] for t in time_uvs)
    assert time_uvs[0] == 25.0


def _make_drift_at_depth_dataset(dt_s: float) -> tuple[xr.Dataset, float, float, float]:
    """Synthetic surface→dive(105 overshoot)→correction→drift(100±2)→climb→surface
    cycle.  Returns (ds, t_dive_start, t_drift_start, t_climb_start) in seconds."""
    t_surface, t_dive, t_overshoot, t_drift, t_climb = (
        120.0,
        1500.0,
        120.0,
        3600.0,
        1500.0,
    )
    t0_dive = t_surface
    t0_overshoot = t0_dive + t_dive
    t0_drift = t0_overshoot + t_overshoot
    t0_climb = t0_drift + t_drift
    t = np.arange(0.0, t0_climb + t_climb + t_surface, dt_s)
    p = np.full(len(t), 0.5)
    m = (t >= t0_dive) & (t < t0_overshoot)
    p[m] = 105.0 * (t[m] - t0_dive) / t_dive
    m = (t >= t0_overshoot) & (t < t0_drift)
    p[m] = 105.0 - 5.0 * (t[m] - t0_overshoot) / t_overshoot
    m = (t >= t0_drift) & (t < t0_climb)
    p[m] = 100.0 + 2.0 * np.sin(2.0 * np.pi * (t[m] - t0_drift) / 300.0)
    m = (t >= t0_climb) & (t < t0_climb + t_climb)
    p[m] = 100.0 * (1.0 - (t[m] - t0_climb) / t_climb)
    return (
        xr.Dataset({"pressure": ("time", p)}, coords={"time": t}),
        t0_dive,
        t0_drift,
        t0_climb,
    )


def _make_state_arrays(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate (state, dive_id, climb_id, profile_id) all initialised to -1."""
    return (
        np.full(n, -1, dtype="b"),
        np.full(n, -1, dtype="i4"),
        np.full(n, -1, dtype="i4"),
        np.full(n, -1, dtype="i4"),
    )


def _set_profile(arrs, slc, state_val, id_val, pid_val):
    """Set state, the matching dive_id or climb_id, and profile_id over slc."""
    state, dive_id, climb_id, profile_id = arrs
    state[slc] = state_val
    (dive_id if state_val == 1 else climb_id)[slc] = id_val
    profile_id[slc] = pid_val


@pytest.mark.parametrize("dt_s", [1.0, 15.0], ids=["post_recovery_1s", "realtime_15s"])
def test_get_profiles_drift_at_depth(dt_s: float) -> None:
    """Drift labelled state=3; post-drift climb spans the full water column
    rather than being mis-attributed to the brief overshoot correction."""
    ds, _, t_drift, t_climb = _make_drift_at_depth_dataset(dt_s)
    result = prof.get_profiles(ds, shallowest_profile=5.0, min_surface_time=180.0)
    state, time = result.state.values, ds.time.values

    drift_state = state[(time >= t_drift) & (time < t_climb)]
    assert (drift_state == 3).mean() > 0.7
    assert (state[time >= t_climb] == 2).mean() > 0.7

    climb_p = ds.pressure.values[state == 2]
    assert climb_p.max() > 90.0 and climb_p.min() < 5.0
    assert 3 in result.state.attrs["flag_values"]


@pytest.mark.parametrize("dt_s", [1.0, 15.0], ids=["post_recovery_1s", "realtime_15s"])
def test_get_profiles_climb_with_stall(dt_s: float) -> None:
    """A 2-minute stall (within the 3-minute tolerance) within the climb must
    not split the climb at either sampling rate."""
    t_surf, t_dive, t_pre, t_stall, t_post = 60.0, 800.0, 600.0, 120.0, 400.0
    t0_dive = t_surf
    t0_pre = t0_dive + t_dive
    t0_stall = t0_pre + t_pre
    t0_post = t0_stall + t_stall
    t0_end = t0_post + t_post
    t = np.arange(0.0, t0_end + t_surf, dt_s)
    p = np.full(len(t), 0.5)
    m = (t >= t0_dive) & (t < t0_pre)
    p[m] = 100.0 * (t[m] - t0_dive) / t_dive
    m = (t >= t0_pre) & (t < t0_stall)
    p[m] = 100.0 - 80.0 * (t[m] - t0_pre) / t_pre
    m = (t >= t0_stall) & (t < t0_post)  # stall: 20 → 22 dbar over 2 min
    p[m] = 20.0 + 2.0 * (t[m] - t0_stall) / t_stall
    m = (t >= t0_post) & (t < t0_end)
    p[m] = 22.0 * (1.0 - (t[m] - t0_post) / t_post)
    p[0] = np.nan  # profinder needs NaN to avoid valid_idx edge case
    ds = xr.Dataset({"pressure": ("time", p)}, coords={"time": t})

    result = prof.get_profiles(ds, shallowest_profile=5.0, min_surface_time=180.0)
    assert (result.state.values[(t >= t0_post) & (t < t0_end)] == 2).mean() > 0.7
    cids = np.unique(result.climb_id.values[result.climb_id.values >= 0])
    assert len(cids) == 1


def test_get_profiles_drift_x_hover_active() -> None:
    """x_hover_active takes precedence over pressure-based detection.  Sparse
    transition reports must be forward-filled and -127 sentinel filtered."""
    ds, _, t_drift, t_climb = _make_drift_at_depth_dataset(15.0)
    time = ds.time.values
    flt = xr.Dataset(
        {
            "m_present_time": ("i", np.array([0.0, 60.0, t_drift, t_climb])),
            "x_hover_active": ("i", np.array([0.0, -127.0, 1.0, 0.0])),
        }
    )
    result = prof.get_profiles(
        ds, shallowest_profile=5.0, min_surface_time=180.0, flt=flt
    )
    state = result.state.values
    assert (state[(time >= t_drift) & (time < t_climb)] == 3).mean() > 0.9
    assert (state[time >= t_climb] == 2).mean() > 0.7


def test_absorb_post_drift_transients() -> None:
    """Brief dive/unknown segments between drift and climb are folded into climb."""
    arrs = _make_state_arrays(50)
    state, dive_id, climb_id, profile_id = arrs
    drift_mask = np.zeros(50, dtype=bool)
    _set_profile(arrs, slice(5, 15), 1, 1, 1)  # main dive
    state[15:30] = 3
    drift_mask[15:30] = True  # drift
    _set_profile(arrs, slice(30, 32), 1, 2, 2)  # spurious dive
    # state[32:34] left as -1 (unknown gap)
    _set_profile(arrs, slice(34, 48), 2, 1, 3)  # real climb

    prof._absorb_post_drift_transients(*arrs, drift_mask, 10.0, 180.0)

    assert np.all(state[30:34] == 2) and np.all(climb_id[30:34] == 1)
    assert np.all(profile_id[30:34] == 3) and np.all(dive_id[30:34] == -1)
    assert (
        np.all(state[5:15] == 1)
        and np.all(state[15:30] == 3)
        and np.all(state[34:48] == 2)
    )


def test_absorb_apex_unknowns_splits_at_max_pressure() -> None:
    """Short unknown gap between dive and climb is split at the pressure max."""
    arrs = _make_state_arrays(30)
    state, dive_id, climb_id, profile_id = arrs
    pressure = np.zeros(30)
    _set_profile(arrs, slice(5, 13), 1, 1, 1)
    pressure[5:13] = np.linspace(10, 80, 8)
    pressure[13:17] = [85, 88, 90, 87]  # apex at index 15
    _set_profile(arrs, slice(17, 25), 2, 1, 2)
    pressure[17:25] = np.linspace(85, 5, 8)

    prof._absorb_apex_unknowns(*arrs, pressure, 10.0, 180.0, 2.0)

    assert np.all(state[13:16] == 1) and np.all(dive_id[13:16] == 1)
    assert np.all(profile_id[13:16] == 1)
    assert state[16] == 2 and climb_id[16] == 1 and profile_id[16] == 2


def test_absorb_apex_unknowns_climb_to_dive_yo_top() -> None:
    """Short underwater gap between climb→dive (yo-top) is split at the
    pressure minimum: pre-min → climb, post-min → dive."""
    arrs = _make_state_arrays(30)
    state, dive_id, climb_id, profile_id = arrs
    pressure = np.zeros(30)
    _set_profile(arrs, slice(5, 13), 2, 1, 1)
    pressure[5:13] = np.linspace(80, 50, 8)
    pressure[13:17] = [48, 45, 47, 50]  # yo-top min at index 14
    _set_profile(arrs, slice(17, 25), 1, 2, 2)
    pressure[17:25] = np.linspace(55, 200, 8)

    prof._absorb_apex_unknowns(*arrs, pressure, 10.0, 180.0, 2.0)

    assert np.all(state[13:15] == 2) and np.all(climb_id[13:15] == 1)
    assert np.all(profile_id[13:15] == 1)
    assert np.all(state[15:17] == 1) and np.all(dive_id[15:17] == 2)
    assert np.all(profile_id[15:17] == 2)


def test_absorb_apex_unknowns_skips_surface_gap() -> None:
    """A gap that goes shallow (glider surfaced) is left for the surface classifier."""
    arrs = _make_state_arrays(30)
    state, _, _, _ = arrs
    pressure = np.zeros(30)
    state[5:10] = 1
    pressure[5:10] = np.linspace(10, 80, 5)
    state[10:13] = 2
    pressure[10:13] = np.linspace(80, 1, 3)
    pressure[13:17] = [0.3, 0.5, 0.4, 0.6]  # shallow unknown gap
    state[17:22] = 1
    pressure[17:22] = np.linspace(1, 60, 5)
    before = state.copy()

    prof._absorb_apex_unknowns(*arrs, pressure, 10.0, 180.0, 2.0)
    assert np.array_equal(state, before)


def test_absorb_pre_drift_transients() -> None:
    """Brief climb/unknown samples between a dive and a drift (overshoot
    recovery) are folded into the drift."""
    arrs = _make_state_arrays(50)
    state, dive_id, climb_id, profile_id = arrs
    drift_mask = np.zeros(50, dtype=bool)
    _set_profile(arrs, slice(5, 15), 1, 1, 1)  # dive (peak overshoot)
    _set_profile(arrs, slice(15, 18), 2, 1, 2)  # brief recovery climb
    state[18:20] = -1  # brief unknown
    state[20:35] = 3
    drift_mask[20:35] = True  # drift
    _set_profile(arrs, slice(35, 45), 2, 2, 3)  # post-drift climb

    prof._absorb_pre_drift_transients(*arrs, drift_mask, 10.0, 180.0)

    # The recovery climb and unknown gap are now part of the drift
    assert np.all(state[15:20] == 3)
    assert np.all(climb_id[15:20] == -1) and np.all(dive_id[15:20] == -1)
    assert np.all(profile_id[15:20] == -1)
    # Surrounding dive and post-drift climb are unchanged
    assert np.all(state[5:15] == 1) and np.all(state[35:45] == 2)


def test_absorb_pre_drift_transients_does_not_swallow_long_climb() -> None:
    """A long climb (>max_transient_duration) before a drift is not absorbed."""
    arrs = _make_state_arrays(60)
    state, _, _, _ = arrs
    drift_mask = np.zeros(60, dtype=bool)
    _set_profile(arrs, slice(0, 10), 1, 1, 1)
    _set_profile(arrs, slice(10, 35), 2, 1, 2)  # 250 s > 180 s
    state[35:50] = 3
    drift_mask[35:50] = True
    before = state.copy()

    prof._absorb_pre_drift_transients(*arrs, drift_mask, 10.0, 180.0)
    assert np.array_equal(state, before)


def test_absorb_post_drift_transients_does_not_swallow_real_dive() -> None:
    """A long dive (>max_transient_duration) between drift and climb is not absorbed."""
    arrs = _make_state_arrays(60)
    state, _, _, _ = arrs
    drift_mask = np.zeros(60, dtype=bool)
    state[5:15] = 3
    drift_mask[5:15] = True
    _set_profile(arrs, slice(15, 35), 1, 1, 1)  # 200 s > 180 s
    _set_profile(arrs, slice(40, 55), 2, 1, 2)
    before = state.copy()

    prof._absorb_post_drift_transients(*arrs, drift_mask, 10.0, 180.0)
    assert np.array_equal(state, before)
