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


def test_find_dive_cycles() -> None:
    """Test that dive cycles are correctly identified from state transitions."""
    # State sequence: surface -> dive -> climb -> surface -> dive -> climb -> surface
    state = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 0],
        dtype="b",
    )

    cycles = pl1._find_dive_cycles(state)

    # Should find 2 dive cycles
    assert len(cycles) == 2, f"Expected 2 cycles, got {len(cycles)}"

    # First cycle: indices 3-9 (dive+climb), surface 9-11
    cycle_start, cycle_end, surf_start, surf_end = cycles[0]
    assert cycle_start == 3
    assert cycle_end == 9
    assert surf_start == 9
    assert surf_end == 11

    # Second cycle: indices 11-15, surface 15-18
    cycle_start, cycle_end, surf_start, surf_end = cycles[1]
    assert cycle_start == 11
    assert cycle_end == 15
    assert surf_start == 15
    assert surf_end == 18


def test_find_dive_cycles_with_unknown_gaps() -> None:
    """Test that unknown states at inflection points don't break cycles."""
    # State with unknown gap between dive and climb (at inflection)
    # surface -> dive -> unknown (inflection) -> climb -> surface
    state = np.array(
        [0, 0, 1, 1, 1, -1, -1, 2, 2, 2, 0, 0],
        dtype="b",
    )

    cycles = pl1._find_dive_cycles(state)

    # Should find 1 cycle (not 2!)
    assert len(cycles) == 1, f"Expected 1 cycle, got {len(cycles)}"

    cycle_start, cycle_end, surf_start, surf_end = cycles[0]
    # Cycle should span the entire dive+unknown+climb section
    assert cycle_start == 2
    assert cycle_end == 10
    assert surf_start == 10
    assert surf_end == 12
