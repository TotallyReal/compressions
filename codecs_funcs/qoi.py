# https://qoiformat.org/
import cProfile as profile
import pstats
from line_profiler import LineProfiler

pr = profile.Profile()
pr.disable()

from codecs_funcs.encoding_functions import BitStreamListEncoder, Encoder, ImageFileHeaderEncoder, HeaderInfo, BitStreamEncoder
import numpy as np
from typing import List, Tuple
from enum import Enum


Chunk = Tuple[str, str] | Tuple[str, str, str] | Tuple[str, str, str, str]

class ChunkOP(str, Enum):
    run = '11'
    index = '00'
    rgb = '11111110'
    rgba = '11111111'
    diff = '01'
    luma = '10'

    def __repr__(self):
         return str(self)

class QOIChunkifier(Encoder[np.ndarray, List[Chunk]]):
    """
    input should be a sequence of r,g,b,a
    """
    def encode(self, rgba_seq: np.ndarray) -> List[Chunk]:
        previous = (0,0,0,255)

        previous_pixels = [0] * 64
        chunks = []

        bin_lookup_2 = [bin(i)[2:].zfill(2) for i in range(2**2)]
        bin_lookup_4 = [bin(i)[2:].zfill(4) for i in range(2**4)]
        bin_lookup_6 = [bin(i)[2:].zfill(6) for i in range(2**6)]
        bin_lookup_8 = [bin(i)[2:].zfill(8) for i in range(2**8)]

        run_length = 0
        for r,g,b,a in rgba_seq:
            r = int(r)
            g = int(g)
            b = int(b)
            a = int(a)
            # Check if this is part of a run of the same color
            if (r,g,b) == previous[:3]:
                run_length += 1
                continue
            while run_length >= 62:
                chunks.append((ChunkOP.run,'111111'))
                run_length -= 62
            if run_length > 0:
                chunks.append((ChunkOP.run, bin_lookup_6[run_length]))
                run_length=0

            # Check if we have seen this color recently
            index_position = (r*3+g*5+b*7+a*11)%64
            if previous_pixels[index_position] == (r,g,b,a):
                chunks.append((ChunkOP.index, bin_lookup_6[index_position]))
                previous = (r,g,b,a)
                continue

            previous_pixels[index_position] = (r, g, b, a)
            diff_r = (r-previous[0]+128)%256-128
            diff_g = (g-previous[1]+128)%256-128
            diff_b = (b-previous[2]+128)%256-128
            diff_a = a-previous[3]
            previous = (r,g,b,a)

            # alpha channel changed
            if diff_a != 0:
                chunks.append((ChunkOP.rgba,
                               bin_lookup_8[r],
                               bin_lookup_8[g],
                               bin_lookup_8[b],
                               bin_lookup_8[a]))
                continue

            # Very close to previous pixel
            if -2<=diff_r<=1 and -2<=diff_g<=1 and -2<=diff_b<=1:
                chunks.append((ChunkOP.diff,
                               bin_lookup_2[diff_r+2],
                               bin_lookup_2[diff_g+2],
                               bin_lookup_2[diff_b+2]))
                continue

            diff_rg = (diff_r - diff_g + 128)%256-120
            diff_bg = (diff_b - diff_g + 128)%256-120
            diff_g += 32
            # Somewhat close to previous pixel
            if 0<=diff_g<=63 and 0<=diff_rg<=15 and 0<=diff_bg<=15:
                chunks.append((ChunkOP.luma,
                               bin_lookup_6[diff_g],
                               bin_lookup_4[diff_rg],
                               bin_lookup_4[diff_bg]))
                continue

            # same alpha, but too different rgb
            chunks.append((ChunkOP.rgb,
                           bin_lookup_8[r],
                           bin_lookup_8[g],
                           bin_lookup_8[b]))


        while run_length >= 62:
            chunks.append((ChunkOP.run,'111111'))
            run_length -= 62
        if run_length > 0:
            chunks.append((ChunkOP.run,bin(run_length)[2:].zfill(6)))
            run_length = 0

        return chunks

    def decode(self, chunks: List[Chunk]) -> np.ndarray:
        previous = (0,0,0,255)
        r = 0
        g = 0
        b = 0
        a = 255

        previous_pixels = [0] * 64
        sequence = []

        for chunk in chunks:
            chunk_type = chunk[0]

            if chunk_type == ChunkOP.run:
                sequence += [previous] * int(chunk[1], 2)
                continue

            if chunk_type == ChunkOP.index:
                index_position = int(chunk[1], 2)
                previous = previous_pixels[index_position]
                sequence.append(previous)
                continue

            a = previous[3]

            if chunk_type == ChunkOP.rgba:
                r = int(chunk[1], 2)
                g = int(chunk[2], 2)
                b = int(chunk[3], 2)
                a = int(chunk[4], 2)

            if chunk_type == ChunkOP.rgb:
                r = int(chunk[1], 2)
                g = int(chunk[2], 2)
                b = int(chunk[3], 2)

            if chunk_type == ChunkOP.diff:
                r = previous[0] + int(chunk[1], 2) - 2
                g = previous[1] + int(chunk[2], 2) - 2
                b = previous[2] + int(chunk[3], 2) - 2

            if chunk_type == ChunkOP.luma:
                g = previous[1] + int(chunk[1], 2) - 32
                diff_g = g - previous[1]
                r = previous[0] + diff_g + int(chunk[2], 2) - 8
                b = previous[2] + diff_g + int(chunk[3], 2) - 8

            previous = (r, g, b, a)
            index_position = (r*3+g*5+b*7+a*11)%64
            previous_pixels[index_position] = previous
            sequence.append(previous)

        return np.array(sequence)


class ChunkToBitstream(BitStreamEncoder[Chunk]):

    def encode(self, chunk: Chunk) -> str:
        return ''.join(chunk)

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[Chunk, int]:

        def chunk_it(chunk_op: ChunkOP, index: int, *index_sep) -> Tuple[Chunk, int]:
            data: List[str] = [bit_stream[index + i : index + j] for i,j in zip(index_sep, index_sep[1:])]
            return (chunk_op, *data), index + index_sep[-1]

        if bit_stream[index] == '0':
            if bit_stream[index+1] == '0':
                return chunk_it(ChunkOP.index, index, 2, 8)
            else:
                return chunk_it(ChunkOP.diff, index, 2, 4, 6, 8)

        # First bit of chunk op is 1
        if bit_stream[index+1] == '0':
            return chunk_it(ChunkOP.luma, index, 2, 8, 12, 16)

        if bit_stream[index+1:index+7]=='111111':
            if bit_stream[index+7] == '0':
                return chunk_it(ChunkOP.rgb, index, 8, 16, 24, 32)
            else:
                return chunk_it(ChunkOP.rgba, index, 8, 16, 24, 32, 40)

        return chunk_it(ChunkOP.run, index, 2, 8)

class ChunksToBitstream(BitStreamEncoder[List[Chunk]]):

    def encode(self, chunks: List[Chunk]) -> str:
        return ''.join([ ''.join(chunk) for chunk in chunks])

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[List[Chunk], int]:
        n = len(bit_stream)
        chunks = []

        def chunk_it(chunk_op: ChunkOP, index: int, *index_sep):
            data: List[str] = [bit_stream[index + i : index + j] for i,j in zip(index_sep, index_sep[1:])]
            chunks.append((chunk_op, *data))
            return index + index_sep[-1]


        while index < n:
            if bit_stream[index] == '0':
                if bit_stream[index+1] == '0':
                    index = chunk_it(ChunkOP.index, index, 2, 8)
                else:
                    index = chunk_it(ChunkOP.diff, index, 2, 4, 6, 8)
                continue

            # First bit of chunk op is 1
            if bit_stream[index+1] == '0':
                index = chunk_it(ChunkOP.luma, index, 2, 8, 12, 16)
                continue

            if bit_stream[index+1:index+7]=='111111':
                if bit_stream[index+7] == '0':
                    index = chunk_it(ChunkOP.rgb, index, 8, 16, 24, 32)
                else:
                    index = chunk_it(ChunkOP.rgba, index, 8, 16, 24, 32, 40)
                continue

            index = chunk_it(ChunkOP.run, index, 2, 8)

        return chunks, index

class QOIEncoder(BitStreamEncoder[np.array]):

    def __init__(self):
        self.header = ImageFileHeaderEncoder()
        self.chunkifier = QOIChunkifier()
        self.to_bitstream = BitStreamListEncoder(ChunkToBitstream())

    def encode(self, image_arr: np.array) -> str:
        h, w, _ = image_arr.shape
        header_bit = self.header.encode(HeaderInfo(name='qoif', width=w, height=h))
        # ignore channel and color space

        chunks = self.chunkifier.encode(image_arr.reshape(-1,4))
        bit_stream = self.to_bitstream.encode(chunks)

        return f'{header_bit}{bit_stream}'


    def decode_bits(self, bit_stream: str, index: int) -> Tuple[np.array, int]:
        header_info, index = self.header.decode_bits(bit_stream, index)
        assert header_info.name == 'qoif'

        chunks, index = self.to_bitstream.decode_bits(bit_stream=bit_stream, index=index)
        sequence = self.chunkifier.decode(chunks)

        return sequence.reshape(header_info.height, header_info.width, 4), index

