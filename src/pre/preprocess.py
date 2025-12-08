import numpy as np
from cv2 import resize
from matplotlib.colors import LogNorm


def _split(data: np.ndarray, k: float) -> tuple[int]:
    """_summary_

    Args:
        data (np.ndarray): _description_
        k (float): _description_

    Returns:
        tuple[int]: _description_
    """
    b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
    b3, b4 = int((1 - k) * data.shape[0]), int((1 - k) * data.shape[1])
    return b1, b2, b3, b4


def map_noise_mean(data: np.ndarray, k: float = 0.1) -> float:
    """_summary_

    Args:
        data (np.ndarray): _description_
        k (float, optional): _description_. Defaults to 0.1.

    Returns:
        float: _description_
    """
    b1, b2, b3, b4 = _split(data, k)
    upper_left = np.mean(data[:b1, :b2].flatten() ** 2)
    upper_right = np.mean(data[b3:, :b2].flatten() ** 2)
    down_left = np.mean(data[:b1, b4:].flatten() ** 2)
    down_right = np.mean(data[b3:, b4:].flatten() ** 2)
    noise = np.mean([upper_left, upper_right, down_left, down_right])
    return np.sqrt(noise)


def map_noise_std(data: np.ndarray, k: float = 0.1) -> float:
    """_summary_

    Args:
        data (np.ndarray): _description_
        k (float, optional): _description_. Defaults to 0.1.

    Returns:
        float: _description_
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
    shape: tuple[int] = (128, 128),
) -> np.ndarray:
    """_summary_

    Args:
        raw_image (np.ndarray): _description_
        shape (tuple[int], optional): _description_. Defaults to (128, 128).

    Returns:
        np.ndarray: _description_
    """
    raw_image = resize(raw_image, shape)
    im = raw_image / raw_image.max()
    return im


def preprocess_lognorm(
    raw_image: np.ndarray,
    shape: tuple[int] = (128, 128),
    std: bool = False,
    raw: bool = True,
) -> np.ndarray:
    """_summary_

    Args:
        raw_image (np.ndarray): _description_
        shape (tuple[int], optional): _description_. Defaults to (128, 128).
        std (bool, optional): _description_. Defaults to False.
        raw (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    if raw:
        im = preprocess(raw_image, shape=shape)
    else:
        assert np.all(raw_image.shape == shape)
        im = raw_image

    if std:
        vmin = min(map_noise_std(im) * 3, im.max())
    else:
        vmin = min(map_noise_mean(im) * 3, im.max())

    lognorm = LogNorm(vmin=vmin, vmax=None, clip=True)
    im_lognorm = lognorm(im)
    im_lognorm = np.ma.getdata(im_lognorm)
    return im_lognorm
