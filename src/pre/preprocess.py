import numpy as np
from cv2 import resize
from matplotlib.colors import LogNorm


def _split(data: np.ndarray, k: float) -> tuple[int, int, int, int]:
    """Get the corner indices of an array.

    Args:
        data (np.ndarray): input array
        k (float): fraction of the array size that defines the corner region

    Returns:
        tuple[int, int, int, int]: four integers (top, left, bottom, right)
            defining the bounds
    """
    b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
    b3, b4 = int((1 - k) * data.shape[0]), int((1 - k) * data.shape[1])
    return b1, b2, b3, b4


def map_noise_mean(data: np.ndarray, k: float = 0.1) -> float:
    """Estimate image noise using the RMS (root mean square) of corner regions.

    The image is split into four corner regions based on the fraction ``k``.
    For each corner, the mean of squared pixel values is computed, and the
    final noise estimate is the square root of the mean of these four values.

    Args:
        data (np.ndarray): input image
        k (float, optional): fraction of the array size that defines the corner region.
            Defaults to 0.1.

    Returns:
        float: estimated noise level based on RMS in corner regions
    """
    b1, b2, b3, b4 = _split(data, k)
    upper_left = np.mean(data[:b1, :b2].flatten() ** 2)
    upper_right = np.mean(data[b3:, :b2].flatten() ** 2)
    down_left = np.mean(data[:b1, b4:].flatten() ** 2)
    down_right = np.mean(data[b3:, b4:].flatten() ** 2)
    noise = np.mean([upper_left, upper_right, down_left, down_right])
    return np.sqrt(noise)


def map_noise_std(data: np.ndarray, k: float = 0.1) -> float:
    """Estimate image noise using the median standard deviation of corner regions.

    The image is split into four corner regions based on the fraction ``k``.
    The standard deviation is computed independently in each corner, and the
    final noise estimate is the median of these four values.

    Args:
        data (np.ndarray): input image
        k (float, optional): fraction of the array size that defines the corner region.
            Defaults to 0.1.

    Returns:
        float: estimated noise level based on corner-region standard deviations
    """
    b1, b2, b3, b4 = _split(data, k)
    upper_left = np.std(data[:b1, :b2])
    upper_right = np.std(data[b3:, :b2])
    down_left = np.std(data[:b1, b4:])
    down_right = np.std(data[b3:, b4:])
    noise = np.median([upper_left, upper_right, down_left, down_right])
    return noise


def preprocess(
    raw_image: np.ndarray,
    shape: tuple[int, int] = (128, 128),
    p: int = 99,
) -> np.ndarray:
    """Resize and rescale an image.

    The image is resized to the target shape and rescaled by the absolute
    ``p``-th percentile value (default 99) for robust normalization.

    Args:
        raw_image (np.ndarray): raw image or map
        shape (tuple[int, int], optional): target shape (height, width) to which
            the image will be resized. Defaults to (128, 128).
        p (int, optional): ``p``-th percentile value. Defaults to 99.

    Returns:
        np.ndarray: resized and rescaled image as float32
    """
    raw_image = resize(raw_image, shape)
    # for better robustness
    pmax_abs = np.percentile(np.abs(raw_image), p)
    if pmax_abs == 0:
        im = raw_image
    else:
        im = raw_image / pmax_abs
    return im.astype(np.float32, copy=False)


def preprocess_lognorm(
    raw_image: np.ndarray,
    shape: tuple[int, int] = (128, 128),
    std: bool = True,
    raw: bool = True,
    p: int = 99,
) -> np.ndarray:
    """Resize, rescale, and apply log normalization to an image.

    The image is optionally preprocessed first, then a LogNorm transform
    is applied with ``vmin`` estimated from noise statistics in the corner
    regions of the image.

    Args:
        raw_image (np.ndarray): raw or preprocessed image
        shape (tuple[int, int], optional): target shape (height, width) to
            which the image will be resized. Defaults to (128, 128).
        std (bool, optional): if True, use ``map_noise_std`` for noise estimation;
            if False, use ``map_noise_mean``. Defaults to True.
        raw (bool, optional): whether the input is a raw image (requires preprocessing)
            or already resized/rescaled. Defaults to True.
        p (int, optional): percentile index used in ``preprocess`` for rescaling.
            Defaults to 99.

    Returns:
        np.ndarray: resized, rescaled, and log-normalized image as float32
    """
    if raw:
        im = preprocess(raw_image, shape=shape, p=p)
    else:
        assert raw_image.shape == shape
        im = raw_image

    if std:
        vmin = min(map_noise_std(im) * 3, im.max())
    else:
        vmin = min(map_noise_mean(im) * 3, im.max())

    lognorm = LogNorm(vmin=vmin, vmax=None, clip=True)
    im_lognorm = lognorm(im)
    im_lognorm = np.ma.getdata(im_lognorm)
    return im_lognorm.astype(np.float32, copy=False)
