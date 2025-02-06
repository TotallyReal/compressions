import numpy as np

from codecs_funcs.qoi import Encoder, ChunksToBitstream, ChunkOP, QOIChunkifier

def test_chuckifier():
    sequence = np.array([
        [218, 211, 192, 255], [218, 211, 192, 255], [229, 222, 203, 255], [226, 219, 200, 255], [201, 194, 175, 255],
        [229, 222, 203, 255], [222, 215, 196, 255], [214, 207, 188, 255], [222, 215, 196, 255], [213, 206, 187, 255],
        [218, 211, 192, 255], [211, 204, 185, 255], [209, 202, 183, 255], [225, 218, 199, 255], [218, 211, 192, 255],
        [214, 207, 188, 255], [208, 201, 182, 255], [209, 202, 183, 255], [227, 220, 201, 255], [217, 210, 191, 255],
        [228, 221, 202, 255], [213, 206, 187, 255], [205, 198, 179, 255], [205, 198, 179, 255], [205, 198, 179, 255]
    ])

    chunks = [(ChunkOP.rgb, '11011010', '11010011', '11000000'), (ChunkOP.run, '000001'),
              (ChunkOP.luma, '101011', '1000', '1000'), (ChunkOP.luma, '011101', '1000', '1000'),
              (ChunkOP.luma, '000111', '1000', '1000'), (ChunkOP.index, '000111'),
              (ChunkOP.luma, '011001', '1000', '1000'), (ChunkOP.luma, '011000', '1000', '1000'),
              (ChunkOP.index, '011110'), (ChunkOP.luma, '010111', '1000', '1000'), (ChunkOP.index, '100010'),
              (ChunkOP.luma, '011001', '1000', '1000'), (ChunkOP.diff, '00', '00', '00'),
              (ChunkOP.luma, '110000', '1000', '1000'), (ChunkOP.index, '100010'), (ChunkOP.index, '100110'),
              (ChunkOP.luma, '011010', '1000', '1000'), (ChunkOP.index, '011011'),
              (ChunkOP.luma, '110010', '1000', '1000'), (ChunkOP.luma, '010110', '1000', '1000'),
              (ChunkOP.luma, '101011', '1000', '1000'), (ChunkOP.index, '010111'),
              (ChunkOP.luma, '011000', '1000', '1000'), (ChunkOP.run, '000010')]

    encoder = QOIChunkifier()

    assert encoder.encode(sequence) == chunks
    assert np.all(encoder.decode(chunks) == sequence)

def test_chunks_to_bitstream():
    chunks = [
        (ChunkOP.run,'111101'),
        (ChunkOP.run,'101101'),
        (ChunkOP.run,'101111'),
        (ChunkOP.rgba,'10011010','10011010','10011010','10011010'),
        (ChunkOP.rgb,'10011010','10011010','10011010'),
        (ChunkOP.index,'011010'),
        (ChunkOP.diff,'01','10','10'),
        (ChunkOP.luma,'011011','0010','1010'),
    ]

    bitstream = \
        '11'+'111101'+\
        '11'+'101101'+\
        '11'+'101111'+\
        '11111111'+'10011010'+'10011010'+'10011010'+'10011010'+\
        '11111110'+'10011010'+'10011010'+'10011010'+\
        '00'+'011010'+\
        '01'+'01'+'10'+'10'+\
        '10'+'011011'+'0010'+'1010'

    encoder = ChunksToBitstream()
    assert encoder.encode(chunks) == bitstream
    assert encoder.decode(bitstream) == chunks