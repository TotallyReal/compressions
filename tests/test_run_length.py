import numpy as np

from codecs_funcs.run_length import RunLengthEncoder, RL_ToBinaryEncoder, RL_ToBitStream

def test_run_length_encoder():
    source = np.array([10, 0, 0, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0 ,0 ,0, 0, 0])
    target = [(0,10), (2,5), (0,3), (18,-2), (1,-1)]

    encoder = RunLengthEncoder(32)
    assert encoder.encode(source) == target
    assert np.all(encoder.decode(target) == source)

def test_rl_to_binary():
    source = [(0,10), (2,5), (0,3), (18,-2), (1,-1)]
    target = [('0000','0100','1010'), ('0010','0011','101'), ('0000','0010','11'), ('1111','0000',''),
              ('0010','0011','010'), ('0001','0010','01'), ('0000','0000','')]

    encoder = RL_ToBinaryEncoder()
    assert encoder.encode(source) == target
    assert encoder.decode(target) == source

def test_rl_to_bitstream():
    source = [('0000','0100','1010'), ('0010','0011','101'), ('0000','0010','11'), ('1111','0000',''),
              ('0010','0011','010'), ('0001','0010','01'), ('0000','0000','')]
    target = '0000010010100010001110100000010111111000000100011010000100100100000000'

    encoder = RL_ToBitStream()
    assert encoder.encode(source) == target
    assert encoder.decode(target) == source