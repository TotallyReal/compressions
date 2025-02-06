import numpy as np
import pytest

from codecs_funcs.encoding_functions import (
    Encoder, Blocker, BlockEncoder, DCTEncoder, ScaleEncoder, ZigZagEncoder, InitialCoefEncoder, RunLengthEncoder,
    RL_ToBinaryEncoder, RL_ToFullBinaryEncoder, DivisibleSizeEncoder, HookEncoder, Source, Target)


@pytest.mark.parametrize('block_size',[1,2,5,8])
def test_blocker(block_size: int):
    encoder = Blocker(block_size=block_size)

    source = np.random.randint(10, size=(10*block_size, 10*block_size))
    assert np.all(encoder.decode(encoder.encode(source)) == source)

    target = np.random.randint(10, size=(10, 10, block_size, block_size))
    assert np.all(encoder.encode(encoder.decode(target)) == target)
