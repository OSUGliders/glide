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
    dp_threshold: float = 3.0,
    dp_dive_reject: float = 10,
) -> tuple[NDArray, NDArray, NDArray]:
    """The number of if statements is a little overwhelming, but this works, sometimes.

    Rejects dives / climbes that don't start or end in a surfacing if they are too short using the
    dp_dive_reject parameter.

    """

    p = np.asarray(p)

    _log.debug("Finding profiles in array of size %i", p.size)

    # States
    unknown = -1
    surface = 0
    diving = 1
    climbing = 2
    state = np.full_like(p, 0, dtype=int)
    dive_ids = np.full_like(p, -1, dtype=int)
    climb_ids = np.full_like(p, -1, dtype=int)

    dive_id = 0
    climb_id = 0
    idx_dive_start = 0
    idx_dive_end = 0
    dp_latest = 0.0  # Pressure difference between last good measurements

    # Determine initial state
    # Could have more logic here...
    if np.isclose(p[0], 0):
        state[0] = surface
    else:
        state[0] = unknown

    # Loop over pressure values
    for i in range(p.size - 1):
        i1 = i + 1
        p0, p1 = p[i], p[i1]

        if np.isfinite(p1) and np.isfinite(p0):
            dp_latest = np.abs(p1 - p0)

        ## State logic ##
        # If NaN, what are our options?
        # Assume at the surface if the last p value was 0
        if np.isnan(p1) & (p0 < p_near_surface):
            state[i1] = surface
        elif np.isnan(p1) & np.isnan(p0) & (state[i] == surface):
            state[i1] = surface
        # If climbing and close to the surface and run into NaN, assume surface
        elif np.isnan(p1) & (p0 < dp_threshold * dp_latest):
            state[i1] = surface
        # Assume still doing what it was doing before, if the last p value > 0
        elif np.isnan(p1) & (p0 > p_near_surface):
            state[i1] = state[i]
        elif np.isnan(p0) & (p1 > p_near_surface):
            state[i1] = state[i]
        # If p1 not NaN, what are our options?
        elif p1 < p_near_surface:
            state[i1] = surface
        elif p1 > p0:
            state[i1] = diving
        elif p1 < p0:
            state[i1] = climbing
        # If p increases and the last p value was NaN and the last state was surface, assume now diving
        elif (p1 > 0) & np.isnan(p0) & (state[i] == surface):
            state[i1] = diving
            # Update previous state if it has a value
        # Consecutive values within machine precision... something odd, but assume we're doing what we were before
        elif np.isclose(p0, p1):
            state[i1] = state[i]
        else:
            raise RuntimeError(
                f"Cannot determine state at i = {i}, p0 = {p0}, p1 = {p1}, state0 = {state[i]}, state1 = {state[i1]}"
            )

        ## Counters ##
        if (state[i1] == diving) & (state[i] != diving):
            dive_id += 1
            idx_dive_start = i1
            idx_dive_end = i1
        if (state[i1] == climbing) & (state[i] != climbing):
            climb_id += 1
            if climb_id != dive_id:
                raise RuntimeError(
                    "Climb number doesn't match most recent dive number."
                )

        if state[i1] == diving:
            dive_ids[i1] = dive_id
            idx_dive_end = i1
        if state[i1] == climbing:
            climb_ids[i1] = climb_id

        ## Counter logic ##
        # If we got to the surface after a dive, there is some problem
        if (state[i1] == surface) and (state[i] == diving):
            _log.warning(
                "Arrived at surface from dive, there could be an issue at index % i", i1
            )
            state[idx_dive_start : idx_dive_end + 1] = unknown
            dive_ids[idx_dive_start : idx_dive_end + 1] = unknown
            dive_id -= 1

    _log.debug("Found %i dives and %i climbs", dive_id, climb_id)

    # Check for mini-dives
    id = contiguous_regions(state == 1)
    p_end = p[id[:, 1] - 1]
    p_start = p[id[:, 0] - 1]
    dp = p_end - p_start
    remove = (dp < dp_dive_reject) & (p_start > p_near_surface)

    _log.debug("Removing %i mini-dives", remove.sum())

    for j in np.argwhere(remove).flatten():
        in_dive_segment = slice(id[j, 0], id[j, 1])
        state[in_dive_segment] = 2
        climb_ids[in_dive_segment] = climb_ids[id[j, 0] - 1]
        dive_ids[in_dive_segment] = -1
        dive_ids[id[j, 0] :] -= 1
        climb_ids[id[j, 1] :] -= 1

    # Check for mini-climbs
    ic = contiguous_regions(state == 2)

    p_end = p[ic[:, 1] - 1]
    p_start = p[ic[:, 0] - 1]
    dp = p_end - p_start
    remove = (dp > -dp_dive_reject) & (p_start > p_near_surface)

    _log.debug("Removing %i mini-climbs", remove.sum())

    for j in np.argwhere(remove).flatten():
        in_climb_segment = slice(ic[j, 0], ic[j, 1])
        state[in_climb_segment] = 1
        climb_ids[in_climb_segment] = -1
        dive_ids[in_climb_segment] = dive_ids[ic[j, 0] - 1]
        dive_ids[ic[j, 1] :] -= 1
        climb_ids[ic[j, 0] :] -= 1

    dive_ids[dive_ids < 1] = -1
    climb_ids[climb_ids < 1] = -1

    return dive_ids, climb_ids, state
