from astropy.io import fits
import numpy as np


def fits2numpy(path2file: str) -> np.ndarray:
    """_Read fits file and get data from primary table._

    Args:
        path2file (str): _path to the fits file._

    Returns:
        np.ndarray: _data from the primary table of the fits file._
    """
    raw_image = fits.getdata(path2file).astype(np.float32)
    # removing excessive dimensions, leaving only height & width
    return raw_image.squeeze()
