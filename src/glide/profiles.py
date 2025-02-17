# Functions to extract profiles from pressure or depth time series

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

_log = logging.getLogger(__name__)


def contiguous_regions(condition: ArrayLike) -> NDArray:
    """Finds the indices of contiguous True regions in a boolean array.

    Parameters
    ----------
    condition : array_like
            Array of boolean values.

    Returns
    -------
    idx : ndarray
            Array of indices demarking the start and end of contiguous True regions in condition.
            Shape is (N, 2) where N is the number of regions.

    """

    condition = np.asarray(condition)
    d = np.diff(condition)
    (idx,) = d.nonzero()
    idx += 1

    if condition[0]:
        idx = np.r_[0, idx]

    if condition[-1]:
        idx = np.r_[idx, condition.size]

    idx.shape = (-1, 2)
    return idx


def find_profiles_using_logic(
    p: ArrayLike,
    p_near_surface: float = 1.0,
    dp_threshold: float = 5.0,
    dp_dive_reject: float = 10,
) -> tuple[NDArray, NDArray, NDArray]:
    """The number of if statements is a little overwhelming, but this works, sometimes.

    Rejects dives / climbs that don't start or end in a surfacing if they are too short using the
    dp_dive_reject parameter.
    """

    p = np.asarray(p)
    _log.debug("Finding profiles in array of size %i", p.size)

    # States
    UNKNOWN, SURFACE, DIVING, CLIMBING = -1, 0, 1, 2
    state = np.full_like(p, SURFACE, dtype=int)
    dive_ids = np.full_like(p, UNKNOWN, dtype=int)
    climb_ids = np.full_like(p, UNKNOWN, dtype=int)

    dive_id, climb_id = 0, 0
    idx_dive_start, idx_dive_end = 0, 0
    dp_latest = 0.0

    def update_state(i, i1, p0, p1):
        nonlocal dp_latest
        if np.isfinite(p1) and np.isfinite(p0):
            dp_latest = np.abs(p1 - p0)

        if np.isnan(p1) and p0 < p_near_surface:
            return SURFACE
        if np.isnan(p1) and np.isnan(p0) and state[i] == SURFACE:
            return SURFACE
        if np.isnan(p1) and p0 < dp_threshold * dp_latest:
            return SURFACE
        if np.isnan(p1) and p0 > p_near_surface:
            return state[i]
        if np.isnan(p0) and p1 > p_near_surface:
            return state[i]
        if p1 < p_near_surface:
            return SURFACE
        if p1 > p0:
            return DIVING
        if p1 < p0:
            return CLIMBING
        if p1 > 0 and np.isnan(p0) and state[i] == SURFACE:
            return DIVING
        if np.isclose(p0, p1):
            return state[i]
        raise RuntimeError(
            f"Cannot determine state at i = {i}, p0 = {p0}, p1 = {p1}, state0 = {state[i]}, state1 = {state[i1]}"
        )

    def handle_mini_profiles(profile_type, state_value, reject_condition):
        regions = contiguous_regions(state == state_value)
        p_end = p[regions[:, 1] - 1]
        p_start = p[regions[:, 0] - 1]
        dp = p_end - p_start
        remove = reject_condition(dp, p_start)

        _log.debug("Removing %i mini-%s", remove.sum(), profile_type)

        for j in np.argwhere(remove).flatten():
            segment = slice(regions[j, 0], regions[j, 1])
            state[segment] = CLIMBING if state_value == DIVING else DIVING
            if state_value == DIVING:
                climb_ids[segment] = climb_ids[regions[j, 0] - 1]
                dive_ids[segment] = UNKNOWN
            else:
                dive_ids[segment] = dive_ids[regions[j, 0] - 1]
                climb_ids[segment] = UNKNOWN
            dive_ids[regions[j, 0] :] -= 1
            climb_ids[regions[j, 1] :] -= 1

    # Determine initial state
    state[0] = SURFACE if np.isclose(p[0], 0) else UNKNOWN

    # Loop over pressure values
    for i in range(p.size - 1):
        i1 = i + 1
        p0, p1 = p[i], p[i1]
        state[i1] = update_state(i, i1, p0, p1)

        if state[i1] == DIVING and state[i] != DIVING:
            dive_id += 1
            idx_dive_start, idx_dive_end = i1, i1
        if state[i1] == CLIMBING and state[i] != CLIMBING:
            climb_id += 1
            if climb_id != dive_id:
                raise RuntimeError(
                    "Climb number doesn't match most recent dive number."
                )

        if state[i1] == DIVING:
            dive_ids[i1] = dive_id
            idx_dive_end = i1
        if state[i1] == CLIMBING:
            climb_ids[i1] = climb_id

        if state[i1] == SURFACE and state[i] == DIVING:
            _log.warning(
                "Arrived at surface from dive, there could be an issue at index %i", i1
            )
            state[idx_dive_start : idx_dive_end + 1] = UNKNOWN
            dive_ids[idx_dive_start : idx_dive_end + 1] = UNKNOWN
            dive_id -= 1

    _log.debug("Found %i dives and %i climbs", dive_id, climb_id)

    handle_mini_profiles(
        "dives",
        DIVING,
        lambda dp, p_start: (dp < dp_dive_reject) & (p_start > p_near_surface),
    )
    handle_mini_profiles(
        "climbs",
        CLIMBING,
        lambda dp, p_start: (dp > -dp_dive_reject) & (p_start > p_near_surface),
    )

    dive_ids[dive_ids < 1] = UNKNOWN
    climb_ids[climb_ids < 1] = UNKNOWN

    return dive_ids, climb_ids, state
