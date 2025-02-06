import itertools
from abc import abstractmethod
from collections import defaultdict
from typing import TypeVar, Generic, Iterator, Callable, NamedTuple
import numpy as np
from scipy import fftpack
from typing import List, Tuple

# <editor-fold desc=" ------------------------ General Encoders ------------------------">

Source = TypeVar('Source')
Target = TypeVar('Target')
SourceInner = TypeVar('SourceInner')
TargetInner = TypeVar('TargetInner')

class Encoder(Generic[Source, Target]):

    @abstractmethod
    def encode(self, source: Source) -> Target:
        pass

    @abstractmethod
    def decode(self, target: Target) -> Source:
        pass

    def __str__(self):
        return self.__class__.__name__

    def at_position(self, index: int):
        return TuplePartialEncoder(self, index)

class PartialEncoder(Encoder[Source, Target]):
    """
    Apply encoder on part of the source \ target
    """

    def __init__(
            self, encoder: Encoder[SourceInner, TargetInner],
            extract_source: Callable[[Source], SourceInner],
            extract_target: Callable[[Target], TargetInner],
            replace_source: Callable[[Source, TargetInner], Target],
            replace_target: Callable[[Target, SourceInner], Source]
    ):
        self.encoder = encoder
        self.extract_source = extract_source
        self.extract_target = extract_target
        self.replace_source = replace_source
        self.replace_target = replace_target

    def encode(self, source: Source) -> Target:
        target_inner = self.encoder.encode(self.extract_source(source))
        return self.replace_source(source, target_inner)

    def decode(self, target: Target) -> Source:
        source_inner = self.encoder.decode(self.extract_target(target))
        return self.replace_target(target, source_inner)

class TuplePartialEncoder(PartialEncoder[Tuple, Tuple]):
    """
    Apply the encoding on a given position in a tuple.
    """

    def __init__(self, encoder: Encoder[SourceInner, TargetInner], index: int):
        super().__init__(encoder,
                         lambda source: source[index], lambda target: target[index],
                         lambda tup, value: tup[:index] + (value,) + tup[index+1:],
                         lambda tup, value: tup[:index] + (value,) + tup[index+1:])
        self.index = index

    def __str__(self):
        return f'{self.encoder} at {self.index}'

class ListEncoder(Encoder[List[Source], List[Target]]):
    """
    Apply the encoder on each element of a list
    """

    def __init__(self, encoder:Encoder[Source, Target]):
        self.encoder = encoder

    def __str__(self):
        return f'List({self.encoder})'

    def encode(self, sources: List[Source]) -> List[Target]:
        return [self.encoder.encode(source) for source in sources]

    def decode(self, targets: List[Target]) -> List[Source]:
        return [self.encoder.decode(target) for target in targets]

class ListCombiner(Encoder[List[List[Source]], List[Source]]):
    """
    Combines several list according to given set of weights.
    For example, for (4,3,1), combines three lists into one by taking
        4 elements from the first,
        3 elements from the second,
        1 element  from the third
    and repeat until taking all the elements.
    """

    def __init__(self, pattern: List[int]):
        assert all(weight > 0 for weight in pattern), "All weights must be positive integers"
        self.pattern = pattern

    def encode(self, lists: List[List[Source]]) -> List[Source]:
        assert len(lists) == len(self.pattern), 'Number of lists does not fit the number of weights'
        divmods = [divmod(len(ll), amount) for ll, amount in zip(lists, self.pattern)]
        assert all(remainder==0 for _, remainder in divmods), 'Each list length should be divisible by the given weight in the pattern'
        assert all(q==divmods[0][0] for q, _ in divmods), 'All list/weight must be the same'

        iterators = [iter(ll) for ll in lists]
        elements = []

        index = 0
        n = len(lists[0])

        while index < n:
            for iterator, weight in zip(iterators, self.pattern):
                for _ in range(weight):
                    elements.append(next(iterator))
            index += self.pattern[0]

        return elements

    def decode(self, target: List[Source]) -> List[List[Source]]:
        lists = [ [] for _ in self.pattern]
        iterator = iter(target)
        n = len(target)
        sum_weights = sum(self.pattern)
        assert n % sum_weights == 0
        index = 0
        while index < n:
            for i, weight in enumerate(self.pattern):
                for _ in range(weight):
                    lists[i].append(next(iterator))
            index += sum_weights

        return lists

class List2DEncoder(Encoder[List[List[Source]], List[List[Target]]]):
    """
    Apply an encoder to every element in a 2D list.
    """

    def __init__(self, encoder: Encoder[Source, Target]):
        self.encoder = encoder

    def __str__(self):
        return f'List2DEncoder({self.encoder})'

    def encode(self, blocks: List[List[Source]]) -> List[List[Target]]:
        return [
            [self.encoder.encode(block) for block in row] for row in blocks
        ]

    def decode(self, blocks: List[List[Target]]) -> List[List[Source]]:
        return [
            [self.encoder.decode(block) for block in row] for row in blocks
        ]

# </editor-fold>

# <editor-fold desc=" ------------------------ BitStream ------------------------">

"""
When decoding a bit stream (represented here as a string over 0/1), use an index to represent the 
last character read from the stream.
Using this index, we can combine several bit stream decoders one after the other.
"""

# TODO: Consider using a (str, int) structure to indicate a bit stream

class BitStreamEncoder(Encoder[Source, str]):

    @abstractmethod
    def encode(self, source: Source) -> str:
        pass

    @abstractmethod
    def decode_bits(self, bit_stream: str, index: int) -> Tuple[Source, int]:
        pass

    def decode(self, bit_stream: str) -> Source:
        data, _ = self.decode_bits(bit_stream, 0)
        return data

    def __str__(self):
        return self.__class__.__name__

class BitStreamListEncoder(BitStreamEncoder[List[Source]]):
    """
    Apply the bit stream encoder to each element in a list to get a combined bitstream.
    """

    def __init__(self, encoder: BitStreamEncoder[Source]):
        self.encoder = encoder
        self.count = -1

    def encode(self, elements: List[Source]) -> str:
        return ''.join([self.encoder.encode(elem) for elem in elements])

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[List[Source], int]:
        elements: List[Source] = []
        count = 0
        n = len(bit_stream)
        while index < n and count != self.count:
            elem, index = self.encoder.decode_bits(bit_stream, index)
            elements.append(elem)
            count += 1

        return elements, index

# </editor-fold>

# <editor-fold desc=" ------------------------ 2D encoders ------------------------">

class DivisibleSizeEncoder(Encoder[np.ndarray, Tuple[np.ndarray, int, int]]):
    """
    Duplicates the last rows and columns so that they would be divisible by the given size
    """

    def __init__(self, size: int):
        self.size = size

    def encode(self, rgb_arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        height, width = rgb_arr.shape[:2]
        new_w = int(np.ceil(width/self.size)*self.size)
        new_h = int(np.ceil(height/self.size)*self.size)
        # Pad rows
        if new_h > height:
            row_pad = np.tile(rgb_arr[-1:, :, :], (new_h - height, 1, 1))
            rgb_arr = np.vstack((rgb_arr, row_pad))

        # Pad columns
        if new_w > width:
            col_pad = np.tile(rgb_arr[:, -1:, :], (1, new_w - width, 1))
            rgb_arr = np.hstack((rgb_arr, col_pad))

        return rgb_arr, width, height

    def decode(self, data: Tuple[np.ndarray, int, int]) -> np.ndarray:
        (arr, width, height) = data
        return arr[:height, :width, :]

class Blocker(Encoder[np.ndarray, np.ndarray]):
    """
    Encodes 2D arrays of size (n*block_size) x (m*block_size) arrays into n x m arrays where
    each entry is a block_size x block_size block.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size

    def encode(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        if h % self.block_size !=0  or h % self.block_size !=0:
            raise Exception('Both width and height must be divisible by the block size')
        return arr.reshape(
            h // self.block_size, self.block_size,
            w // self.block_size, self.block_size).swapaxes(1, 2)

    def decode(self, arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr) # TODO: Fix this...
        h, w, _, _ = arr.shape
        return arr.swapaxes(1, 2).reshape(h * self.block_size, w * self.block_size)

class Flatten(Encoder[List[List[Source]], Tuple[List[Source], int, int]]):

    def encode(self, source: List[List[Source]]) -> Tuple[List[Source], int, int]:
        rows = len(source)
        cols = len(source[0])
        return [elem for row in source for elem in row], rows, cols

    def decode(self, target: Tuple[List[Source], int, int]) -> List[List[Source]]:
        elements, rows, cols = target
        return [elements[i*cols: (i+1)*cols] for i in range(rows)]

class ScaleEncoder(Encoder[np.ndarray, np.ndarray]):

    def __init__(self, matrix_scale: np.ndarray):
        self.matrix_scale = matrix_scale

    def encode(self, arr: np.ndarray) -> np.ndarray:
        return np.round(arr / self.matrix_scale).astype(dtype=int) # not //   !!!!

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.matrix_scale

class AverageEncoder(Encoder[np.ndarray, np.ndarray]):
    """
    Averages blocks of size block_size x block_size into a single (rounded) integers
    """

    def __init__(self, block_size:int):
        self.block_size = block_size

    def encode(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        # Ensure the array dimensions are divisible by block_size
        assert h % self.block_size == 0 and w % self.block_size == 0, \
            "Array dimensions must be divisible by block_size"

        # Reshape and average
        new_h = h // self.block_size
        new_w = w // self.block_size
        reshaped = arr.reshape(new_h, self.block_size, new_w, self.block_size)
        averaged = np.round(reshaped.mean(axis=(1, 3)))

        return averaged

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(arr, self.block_size, axis=0), self.block_size, axis=1)

class DCTEncoder(Encoder[np.ndarray, np.ndarray]):
    """
    Apply a 2D discrete cosine transform
    """

    def encode(self, block: np.ndarray) -> np.ndarray:
        """Compute the 2D Discrete Cosine Transform (DCT-II)."""
        return fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    def decode(self, block: np.ndarray) -> np.ndarray:
        """Compute the 2D Inverse Discrete Cosine Transform (IDCT-II)."""
        return fftpack.idct(fftpack.idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

class ZigZagEncoder(Encoder[np.ndarray, np.ndarray]):
    """
    Transform a block of elements into a sequence using a zig zag pattern. Namely, move over diagonal (on x=y direction)
    and in each step change the direction of advancement.
    For example
    1,   2 , 3,  4
    5,   6,  7,  8
    9,  10, 11, 12
    13, 14, 15, 16

    becomes

    1, 2, 5, 9, 6, 3, 4, 7, 10, 13, 14, 11, 8, 12, 15, 16
    """

    def __init__(self, block_size: int):
        self.block_size = block_size

    def zig_zag_indices(self) -> Iterator[Tuple[int, int]]:
        even = False
        for diag_len in range(1, self.block_size + 1):
            indices = range(diag_len) if even else range(diag_len - 1, -1, -1)
            for i in indices:
                yield i, diag_len - i - 1
            even = not even

        for diag_len in range(self.block_size + 1, 2 * self.block_size):
            indices = range(2 * self.block_size - diag_len) if even else range(2 * self.block_size - diag_len - 1, -1, -1)
            for i in indices:
                yield diag_len - self.block_size + i, self.block_size - i - 1
            even = not even

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        sequence = np.zeros(shape=(self.block_size * self.block_size), dtype=matrix.dtype)

        for index, (row, col) in enumerate(self.zig_zag_indices()):
            sequence[index] = matrix[row, col]
        return sequence

    def decode(self, sequence: np.ndarray) -> np.ndarray:
        matrix = np.zeros(shape=(self.block_size, self.block_size), dtype=sequence.dtype)

        for index, (row, col) in enumerate(self.zig_zag_indices()):
            matrix[row, col] = sequence[index]
        return matrix

# </editor-fold>

# <editor-fold desc=" ------------------------ Image Encoders ------------------------">

HeaderInfo = NamedTuple('HeaderInfo', name=str, width=int, height=int)

class ImageFileHeaderEncoder(BitStreamEncoder[HeaderInfo]):
    """
    Uses 96 bits (12 bytes) to write header name, width and height
    """
    def encode(self, source: HeaderInfo) -> str:
        assert len(source.name) == 4
        header_name_bit = ''.join(format(ord(c), '08b') for c in source.name)
        width_bit = bin(source.width)[2:].zfill(32)
        height_bit = bin(source.height)[2:].zfill(32)
        return f'{header_name_bit}{width_bit}{height_bit}'

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[HeaderInfo, int]:
        header_name = ''.join(chr(int(bit_stream[i:i+8], 2)) for i in range(index, index+32, 8))
        width = int(bit_stream[index+32:index+64], 2)
        height = int(bit_stream[index+64:index+96], 2)
        return HeaderInfo(name=header_name, width=width, height=height), index+96

YCbCr = Tuple[np.ndarray, np.ndarray, np.ndarray]

class YCbCrEncoder(Encoder[np.ndarray, YCbCr]):

    def __init__(self):
        # Transformation matrix for YCbCr , using BT.601
        # Note that:
        #     Y  = 0.299*R + 0.587*G + 0.114*B
        #     Cb = 0.564*(B-Y)
        #     Cr = 0.713*(R-Y)
        self.to_ycbcr_mat = np.array([
            [0.299, 0.587, 0.114],
            [-0.1687, -0.3313, 0.5],
            [0.5, -0.4187, -0.0813]
        ])
        self.to_ycbcr_offset = np.array([0, 128, 128])

        # YCbCr to RGB transformation matrix
        self.to_rgb_mat = np.array([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ])

        # Offset for Cb and Cr channels (128 for digital YCbCr)
        self.to_rgb_offset = np.array([0, 128, 128])

    def encode(self, rgb_arr: np.ndarray) -> YCbCr:
        ycbcr_arr = np.dot(rgb_arr, self.to_ycbcr_mat.T) + self.to_ycbcr_offset

        return ycbcr_arr[..., 0], ycbcr_arr[..., 1], ycbcr_arr[..., 2]

    def decode(self, ycbcr: YCbCr) -> np.ndarray:
        ycbcr_arr = np.dstack((ycbcr[0], ycbcr[1], ycbcr[2]))
        ycbcr_arr = ycbcr_arr.astype(np.float32)
        ycbcr_arr -= self.to_rgb_offset
        rgb_arr = np.dot(ycbcr_arr, self.to_rgb_mat.T)
        rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)

        return rgb_arr

# </editor-fold>

# <editor-fold desc=" ------------------------ Run Length encoding ------------------------">



# </editor-fold>

class InitialCoefEncoder(Encoder[np.ndarray, np.ndarray]):

    # Changes the blocks variable

    def encode(self, blocks: np.ndarray) -> np.ndarray:
        rows = len(blocks)
        cols = len(blocks[0])
        for row in range(rows-1,0,-1):
            for col in range(cols-1,0,-1):
                blocks[row][col][0] -= int( (blocks[row-1][col][0]+blocks[row-1][col-1][0]+blocks[row][col-1][0])//3 )

        for row in range(rows - 1, 0, -1):
            blocks[row][0][0] -= blocks[row - 1][0][0]

        for col in range(cols - 1, 0, -1):
            blocks[0][col][0] -= blocks[0][col - 1][0]

        return blocks

    def decode(self, blocks: np.ndarray) -> np.ndarray:
        rows = len(blocks)
        cols = len(blocks[0])
        for row in range(1, rows):
            blocks[row][0][0] += blocks[row - 1][0][0]

        for col in range(1, cols):
            blocks[0][col][0] += blocks[0][col - 1][0]

        for row in range(1, rows):
            for col in range(1, cols):
                blocks[row][col][0] += int( (blocks[row-1][col][0]+blocks[row-1][col-1][0]+blocks[row][col-1][0])//3 )

        return blocks

class HookEncoder(Encoder[Source, Target]):

    def __init__(self, encoder: Encoder[Source, Target]):
        self.encoder = encoder
        self.last_encode_input = None
        self.last_decode_input = None
        self.last_encode_output = None
        self.last_decode_output = None

    def encode(self, source: Source) -> Target:
        self.last_encode_input = source
        self.last_encode_output = self.encoder.encode(source)
        return self.last_encode_output

    def decode(self, target: Target) -> Source:
        self.last_decode_input = target
        self.last_decode_output = self.encoder.decode(target)
        return self.last_decode_output


