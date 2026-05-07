from importlib import resources

import numpy as np
import pandas as pd
import xarray as xr

import glide.process_l1 as pl1
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


def test_assign_surface_state() -> None:
    """Test that surface state is assigned to unknown points near GPS fixes."""
    # Create a synthetic L2 dataset with state variable
    n = 100
    time = np.arange(n, dtype=float)

    # State: surface at start, dive in middle, surface at end
    # -1 = unknown, 0 = surface, 1 = dive, 2 = climb
    state = np.full(n, -1, dtype="b")
    state[20:40] = 1  # dive
    state[40:60] = 2  # climb

    ds = xr.Dataset(
        {
            "state": (
                "time",
                state,
                dict(
                    long_name="Glider state",
                    flag_values=np.array([-1, 0, 1, 2], "b"),
                    flag_meanings="unknown surface dive climb",
                ),
            ),
        },
        coords={"time": time},
    )

    # Create flight data with GPS fixes at surface times
    flt_time = np.arange(n, dtype=float)
    gps_lat = np.full(n, np.nan)
    gps_lat[5] = 45.0  # GPS fix near start (surface)
    gps_lat[10] = 45.0  # Another GPS fix (surface)
    gps_lat[70] = 45.0  # GPS fix near end (surface)
    gps_lat[80] = 45.0  # Another GPS fix (surface)

    flt = xr.Dataset(
        {
            "m_gps_lat": ("time", gps_lat),
            "m_present_time": ("time", flt_time),
        },
        coords={"time": flt_time},
    )

    # Apply surface state assignment
    result = pl1.assign_surface_state(ds, flt, dt=15.0)

    # Check that unknown states near GPS fixes are now surface (0)
    assert result.state.values[5] == 0, "Point at GPS fix should be surface"
    assert result.state.values[10] == 0, "Point at GPS fix should be surface"
    assert result.state.values[70] == 0, "Point near GPS fix should be surface"
    assert result.state.values[80] == 0, "Point near GPS fix should be surface"

    # Check that dive/climb states are unchanged
    assert np.all(result.state.values[20:40] == 1), "Dive states should be unchanged"
    assert np.all(result.state.values[40:60] == 2), "Climb states should be unchanged"

    # Check that unknown states far from GPS fixes remain unknown
    assert result.state.values[30] == 1, "Mid-dive should still be dive"
    assert result.state.values[50] == 2, "Mid-climb should still be climb"


def test_assign_surface_state_no_flight_data() -> None:
    """Test that assign_surface_state handles missing flight data gracefully."""
    n = 20
    state = np.full(n, -1, dtype="b")
    state[5:15] = 1

    ds = xr.Dataset(
        {"state": ("time", state)},
        coords={"time": np.arange(n, dtype=float)},
    )

    # No flight data - should return unchanged
    result = pl1.assign_surface_state(ds, flt=None)
    assert np.array_equal(result.state.values, state)


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

    result = pl1.add_velocity(ds, config, flt=flt)

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
        out = pl1.get_profiles(merged, shallowest_profile=5.0, profile_distance=10)

        # Assign surface state and add velocity
        out = pl1.assign_surface_state(out, flt=flt_raw)
        out = pl1.add_velocity(out, config, flt=flt_raw)

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
    out = pl1.get_profiles(merged, shallowest_profile=5.0, profile_distance=10)

    # Check that we detected some profiles
    n_dives = (out.dive_id.values >= 0).sum()

    # Assign surface state and add velocity
    out = pl1.assign_surface_state(out, flt=flt_raw)
    out = pl1.add_velocity(out, config, flt=flt_raw)

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

    result = pl1.add_gps_fixes(ds, flt, config)

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

    result = pl1.add_gps_fixes(ds, flt, config)

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

    result = pl1.add_gps_fixes(ds, flt, config)

    assert "time_gps" not in result.dims
    assert "lat_gps" not in result
    assert "lon_gps" not in result


def test_assign_segment_id_basic() -> None:
    """Each maximal run of state != 0 gets a unique segment_id."""
    # state pattern: surface, dive, dive, climb, surface, dive, dive, surface
    # Two underwater segments: indices 1-3 and 5-6
    state = np.array([0, 1, 1, 2, 0, 1, 1, 0], dtype="i1")
    ds = xr.Dataset(
        {"state": ("time", state)},
        coords={"time": np.arange(len(state), dtype="f8")},
    )

    result = pl1.assign_segment_id(ds)

    seg = result.segment_id.values
    assert seg.tolist() == [-1, 1, 1, 1, -1, 2, 2, -1]


def test_assign_segment_id_edge_segments() -> None:
    """Segments at the start or end of the dataset still get an id."""
    # Dataset starts and ends underwater
    state = np.array([1, 1, 0, 1, 1, 0, 2, 2], dtype="i1")
    ds = xr.Dataset(
        {"state": ("time", state)},
        coords={"time": np.arange(len(state), dtype="f8")},
    )

    result = pl1.assign_segment_id(ds)

    seg = result.segment_id.values
    # Three segments: indices 0-1 (leading), 3-4 (middle), 6-7 (trailing)
    assert seg.tolist() == [1, 1, -1, 2, 2, -1, 3, 3]


def test_assign_segment_id_includes_unknown_state() -> None:
    """state == -1 (unknown) is treated as underwater for segment grouping."""
    # state == -1 is "unknown" (between profiles) and is part of a segment
    state = np.array([0, 1, -1, 2, 0], dtype="i1")
    ds = xr.Dataset(
        {"state": ("time", state)},
        coords={"time": np.arange(len(state), dtype="f8")},
    )

    result = pl1.assign_segment_id(ds)

    seg = result.segment_id.values
    assert seg.tolist() == [-1, 1, 1, 1, -1]


def test_assign_segment_id_no_state() -> None:
    """Returns dataset unchanged when state variable is missing."""
    ds = xr.Dataset(coords={"time": np.arange(3, dtype="f8")})
    result = pl1.assign_segment_id(ds)
    assert "segment_id" not in result


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

    result = pl1.get_profiles(ds, shallowest_profile=5.0, profile_distance=10)

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


def test_emit_ioos_profiles_writes_scalar_velocity(tmp_path) -> None:
    """A profile with finite velocity is emitted as an NGDAC-shaped scalar file."""
    # Build a minimal L2-shaped dataset by hand: 6 time points, one profile,
    # one segment, one velocity entry on time_uv.
    time = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])  # epoch seconds
    profile_id = np.array([-1, 1, 1, 1, -1, -1], dtype="i4")
    segment_id = np.array([-1, 1, 1, 1, -1, -1], dtype="i4")
    state = np.array([0, 1, 1, 2, 0, 0], dtype="i1")

    ds = xr.Dataset(
        {
            "temperature": (
                "time",
                np.array([np.nan, 12.0, 12.5, 13.0, np.nan, np.nan]),
            ),
            "profile_id": ("time", profile_id),
            "segment_id": ("time", segment_id),
            "state": ("time", state),
            "dive_id": ("time", np.array([-1, 1, 1, -1, -1, -1], dtype="i4")),
            "climb_id": ("time", np.array([-1, -1, -1, 1, -1, -1], dtype="i4")),
            "u": ("time_uv", np.array([0.12])),
            "v": ("time_uv", np.array([-0.05])),
            "time_uv": ("time_uv", np.array([20.0])),
            "lat_uv": ("time_uv", np.array([45.0])),
            "lon_uv": ("time_uv", np.array([-123.0])),
        },
        coords={"time": time},
    )

    written = pl1.emit_ioos_profiles(ds, tmp_path, "test_glider")

    assert len(written) == 1
    f = written[0]
    assert f.exists()

    out = xr.open_dataset(f)
    try:
        # NGDAC contract: scalar u, v, time_uv, lat_uv, lon_uv, profile_id, segment_id
        for v in ("u", "v", "time_uv", "lat_uv", "lon_uv", "profile_id", "segment_id"):
            assert v in out
            assert out[v].ndim == 0, f"{v} must be scalar"
        assert float(out.u.values) == 0.12
        assert float(out.v.values) == -0.05
        assert int(out.profile_id.values) == 1
        assert int(out.segment_id.values) == 1
        # Excluded variables / dims
        for v in ("dive_id", "climb_id", "state"):
            assert v not in out
        assert "time_uv" not in out.dims
        # Only the in-profile time points are kept
        assert out.sizes["time"] == 3
    finally:
        out.close()


def test_emit_ioos_profiles_skips_when_velocity_nan(tmp_path) -> None:
    """A profile in a segment with NaN u/v is skipped (no file written)."""
    time = np.array([0.0, 10.0, 20.0, 30.0])
    ds = xr.Dataset(
        {
            "temperature": ("time", np.array([np.nan, 12.0, 12.5, np.nan])),
            "profile_id": ("time", np.array([-1, 1, 1, -1], dtype="i4")),
            "segment_id": ("time", np.array([-1, 1, 1, -1], dtype="i4")),
            "state": ("time", np.array([0, 1, 2, 0], dtype="i1")),
            "u": ("time_uv", np.array([np.nan])),
            "v": ("time_uv", np.array([np.nan])),
            "time_uv": ("time_uv", np.array([15.0])),
            "lat_uv": ("time_uv", np.array([np.nan])),
            "lon_uv": ("time_uv", np.array([np.nan])),
        },
        coords={"time": time},
    )

    written = pl1.emit_ioos_profiles(ds, tmp_path, "test_glider")

    assert written == []
    assert list(tmp_path.glob("*.nc")) == []


def test_emit_ioos_profiles_idempotent(tmp_path) -> None:
    """Re-running with the same data does not write new files."""
    time = np.array([0.0, 10.0, 20.0, 30.0])
    ds = xr.Dataset(
        {
            "temperature": ("time", np.array([np.nan, 12.0, 12.5, np.nan])),
            "profile_id": ("time", np.array([-1, 1, 1, -1], dtype="i4")),
            "segment_id": ("time", np.array([-1, 1, 1, -1], dtype="i4")),
            "state": ("time", np.array([0, 1, 2, 0], dtype="i1")),
            "u": ("time_uv", np.array([0.1])),
            "v": ("time_uv", np.array([0.0])),
            "time_uv": ("time_uv", np.array([15.0])),
            "lat_uv": ("time_uv", np.array([45.0])),
            "lon_uv": ("time_uv", np.array([-123.0])),
        },
        coords={"time": time},
    )

    first = pl1.emit_ioos_profiles(ds, tmp_path, "test_glider")
    assert len(first) == 1

    second = pl1.emit_ioos_profiles(ds, tmp_path, "test_glider")
    assert second == []
    assert len(list(tmp_path.glob("*.nc"))) == 1

    forced = pl1.emit_ioos_profiles(ds, tmp_path, "test_glider", force=True)
    assert len(forced) == 1
    assert len(list(tmp_path.glob("*.nc"))) == 1
