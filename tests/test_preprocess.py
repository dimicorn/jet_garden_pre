import pytest
from pytest import approx
import numpy as np
from pre.preprocess import _split, map_noise_mean, preprocess, preprocess_lognorm


CASES = [
    ("10x10", np.arange(100).reshape(10, 10)),
    ("11x11", np.arange(121).reshape(11, 11)),
    ("20x20", np.arange(400).reshape(20, 20)),
]

EXPECTED_SPLIT = {
    0.1: {
        "10x10": (1, 1, 9, 9),
        "11x11": (1, 1, 9, 9),
        "20x20": (2, 2, 18, 18),
    },
    0.2: {
        "10x10": (2, 2, 8, 8),
        "11x11": (2, 2, 8, 8),
        "20x20": (4, 4, 16, 16),
    },
}

EXPECTED_MAP_NOISE_MEAN = {
    0.1: {
        "10x10": 67.04848991588104,
        "11x11": 77.56851809851726,
        "20x20": 269.0381013908625,
    },
    0.2: {
        "10x10": 63.96483408873973,
        "11x11": 74.21955043428022,
        "20x20": 256.83749726237403,
    },
}

SHAPE = (128, 128)
PERCENTS = [0.1, 0.2]


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.mark.parametrize("percent", PERCENTS)
@pytest.mark.parametrize("name,data", CASES, ids=[c[0] for c in CASES])
def test_split(name, data, percent):
    expected = EXPECTED_SPLIT[percent][name]
    assert _split(data, percent) == expected


@pytest.mark.parametrize("percent", PERCENTS)
@pytest.mark.parametrize("name,data", CASES, ids=[c[0] for c in CASES])
def test_map_noise_mean(name, data, percent):
    expected = EXPECTED_MAP_NOISE_MEAN[percent][name]
    assert map_noise_mean(data, percent) == approx(expected)


@pytest.mark.skip(reason="test_map_noise_std not implemented yet")
def test_map_noise_std():
    # TODO: Add tests
    ...


def test_preprocess(rng: np.random.Generator):  # pylint: disable=redefined-outer-name
    # TODO: Add more tests?
    img_raw = rng.random((512, 512), dtype=np.float32) - 0.5
    img_preprocessed = preprocess(img_raw, shape=SHAPE)
    assert isinstance(img_preprocessed, np.ndarray)
    assert img_preprocessed.dtype == np.float32
    assert img_preprocessed.shape == SHAPE


def test_preprocess_lognorm(rng: np.random.Generator):  # pylint: disable=redefined-outer-name
    # TODO: Add more tests?
    img_raw = rng.random((512, 512), dtype=np.float32) - 0.5
    img_preprocessed = preprocess_lognorm(img_raw, shape=SHAPE)
    assert isinstance(img_preprocessed, np.ndarray)
    assert img_preprocessed.dtype == np.float32
    assert img_preprocessed.shape == SHAPE
