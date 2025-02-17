# Functions for converting variables in different units and formats.

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dpm_to_dd(x: ArrayLike) -> NDArray:
    """Convert lat/lon from degrees append decimal minutes (e.g. 14234.6 is 142 E 34.6') to decimal degress."""
    d = np.trunc(np.asarray(x) / 100)
    m = (x - d * 100) / 60
    return d + m


def bar_to_dbar(x: ArrayLike) -> NDArray:
    """Convert pressure in bar to decibar."""
    return 10 * np.asarray(x)


def spm_to_mspcm(x: ArrayLike) -> NDArray:
    """Convert conductivity in S/m to mS/cm."""
    return 10 * np.asarray(x)


def rad_to_deg(x: ArrayLike) -> NDArray:
    """Convert radians to degrees."""
    return np.rad2deg(x)


def mid(x: ArrayLike) -> NDArray:
    """Estimate mid points of an array. Works with datetimes."""
    x = np.asarray(x)
    return x[:-1] + 0.5 * (x[1:] - x[:-1])
