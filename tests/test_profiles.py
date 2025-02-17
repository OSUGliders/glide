import numpy as np

import glide.profiles as pfls


def test_contiguous_regions() -> None:
    x = [True, True, True, False, True, True]
    idxs = pfls.contiguous_regions(x)
    assert np.all(idxs == np.array([[0, 3], [4, 6]]))


def test_find_profiles_using_logic() -> None:
    p = [0, 0, np.nan, 0, 1, 5, 10, 20, 18, 12, 6, 2, np.nan, 0]
    dive_numbers, climb_numbers, state = pfls.find_profiles_using_logic(p)
    assert np.all(state == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0])
    assert np.all(dive_numbers == [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    assert np.all(climb_numbers == [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1])

    p = [0, 5, 10, 15, 20, 25, 30, 35, 30, 25, 20, 23, 25, 22, 15, 9, 5, 3, 0, 0]
    dive_numbers, climb_numbers, state = pfls.find_profiles_using_logic(p)
    assert np.all(state == [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0])
    assert np.all(
        dive_numbers
        == [-1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    assert np.all(
        climb_numbers
        == [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1]
    )
