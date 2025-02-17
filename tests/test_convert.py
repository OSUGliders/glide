import numpy as np

import glide.convert as conv


def test_dpm_to_dd() -> None:
    lat = 4355.1
    assert np.isclose(conv.dpm_to_dd(lat), 43 + 55.1 / 60)


def test_bar_to_dbar() -> None:
    p = 10
    assert np.isclose(conv.bar_to_dbar(p), 100)


def test_spm_to_mspcm() -> None:
    C = 3
    assert np.isclose(conv.spm_to_mspcm(C), 30)


def test_mid() -> None:
    x = [1, 2, 4, 11]
    assert np.isclose(conv.mid(x), [1.5, 3, 7.5]).all()
