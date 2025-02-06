from pickletools import uint8

from codecs_funcs.filters2D import LeftDiff, UpDiff, UpLeftDiff, PaethDiff, Encoder
import numpy as np

def validate_encoder(encoder: Encoder[np.ndarray, np.ndarray], source: np.ndarray, target: np.ndarray):
    assert np.all(encoder.encode(source) == target)
    assert np.all(encoder.decode(target) == source)

    source_8 = (source % 256).astype(np.uint8)
    target_8 = (target % 256).astype(np.uint8)
    assert np.all(encoder.encode(source_8) == target_8)
    assert np.all(encoder.decode(target_8) == source_8)

def test_left_diff():
    source = np.array([
        [5,2,3],
        [4,7,-1],
        [3,-3,3]
    ])
    target = np.array([
        [5,-3,1],
        [4,3,-8],
        [3,-6,6]
    ])

    validate_encoder(LeftDiff(), source, target)

def test_up_diff():
    source = np.array([
        [5, 2, 3],
        [4, 7, -1],
        [3, -3, 3]
    ]).T
    target = np.array([
        [5, -3, 1],
        [4, 3, -8],
        [3, -6, 6]
    ]).T

    validate_encoder(UpDiff(), source, target)

def test_avg_diff():
    source = np.array([
        [5, 2, 3],
        [4, 6, -1],
        [3, -3, 3]
    ]).T
    target = np.array([
        [5, 2, 3],
        [4, 3, -5],
        [3, -7, 5]
    ]).T

    validate_encoder(UpLeftDiff(), source, target)

def test_paeth_diff():
    source = np.array([
        [5, 2, 3],
        [4, 6, 3],
        [3, 7, 3]
    ]).T
    target = np.array([
        [5, 2, 3],
        [4, 1, -3],
        [3, 1, -4]
    ]).T

    validate_encoder(PaethDiff(), source, target)