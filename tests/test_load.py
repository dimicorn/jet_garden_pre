import numpy as np
from pre.load import fits2numpy

FITS_FILE = "tests/J1608-1625_S_2018_07_01_pet_map.fits"


def test_fits2numpy_basic():
    res = fits2numpy(FITS_FILE)
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.float32
    assert res.shape == (512, 512)
    assert np.isfinite(res).all()
    assert res.ndim == 2


def test_fits2numpy_deterministic():
    res1 = fits2numpy(FITS_FILE)
    res2 = fits2numpy(FITS_FILE)
    assert np.array_equal(res1, res2)
