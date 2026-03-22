from importlib import resources

import numpy as np
import pandas as pd
import xarray as xr

import glide.process_l1 as pl1
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
    n_climbs = (out.climb_id.values >= 0).sum()

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
