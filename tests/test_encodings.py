import numpy as np
import pytest

from codecs_funcs.encoding_functions import Blocker, ZigZagEncoder


@pytest.mark.parametrize('block_size',[1,2,5,8])
def test_blocker(block_size: int):
    encoder = Blocker(block_size=block_size)

    source = np.random.randint(10, size=(10*block_size, 10*block_size))
    assert np.all(encoder.decode(encoder.encode(source)) == source)

    target = np.random.randint(10, size=(10, 10, block_size, block_size))
    assert np.all(encoder.encode(encoder.decode(target)) == target)

def test_zig_zag_encoder():
    source = np.array([
        [1,   2 , 3,  4],
        [5,   6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ])
    target = np.array([1, 2, 5, 9, 6, 3, 4, 7, 10, 13, 14, 11, 8, 12, 15, 16])


    encoder = ZigZagEncoder(4)
    assert np.all(encoder.encode(source) == target)
    assert np.all(encoder.decode(target) == source)
