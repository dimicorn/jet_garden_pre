import numpy as np
from pre.preprocess import (
    _split,
)


def test_split():
    data1 = np.arange(100).reshape((10, 10))
    data2 = np.arange(121).reshape((11, 11))
    assert _split(data1, 0.1) == (1, 1, 9, 9)
    assert _split(data2, 0.1) == (1, 1, 9, 9)


def test_map_noise_mean(): ...


def test_map_noise_std(): ...


def test_preprocess(): ...


def test_preprocess_lognorm(): ...
