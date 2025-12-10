from astropy.io import fits
import numpy as np


def fits2numpy(path2file: str) -> np.ndarray:
    """Read fits file and get data from primary table.

    Args:
        path2file (str): path to the fits file.

    Returns:
        np.ndarray: data from the primary table of the fits file.
    """
    raw_image = fits.getdata(path2file).astype(np.float32)
    # removing excessive dimensions, leaving only height & width
    return raw_image.squeeze()
