import numpy as np
import cv2 as cv
from matplotlib.colors import LogNorm


def map_noise(data: np.ndarray, k: float = 0.1) -> float:
    b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
    b3, b4 = int((1 - k) * data.shape[0]), int((1 - k) * data.shape[1])

    upper_left = np.std(data[:b1, :b2])
    upper_right = np.std(data[b3:, :b2])
    down_left = np.std(data[:b1, b4:])
    down_right = np.std(data[b3:, b4:])
    noise = np.median([upper_left, upper_right, down_left, down_right])
    return noise


def preprocess(raw_image: np.ndarray, shape: tuple[int] = (128, 128)) -> np.ndarray:
    raw_image = cv.resize(raw_image, shape)
    im = raw_image / raw_image.max()
    return im


def preprocess_lognorm(
    raw_image: np.ndarray, shape: tuple[int] = (128, 128)
) -> np.ndarray:
    im = preprocess(raw_image, shape=shape)
    vmin = min(map_noise(im) * 3, im.max())
    lognorm = LogNorm(vmin=vmin, vmax=None, clip=True)
    im_lognorm = lognorm(im)
    im_lognorm = np.ma.getdata(im_lognorm)
    return im_lognorm
