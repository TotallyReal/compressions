from abc import abstractmethod
from collections import defaultdict
from typing import TypeVar, Generic, Iterator
import numpy as np
from scipy import fftpack
from typing import List, Tuple

Source = TypeVar('Source')
Target = TypeVar('Target')


class Encoder(Generic[Source, Target]):

    @abstractmethod
    def encode(self, source: Source) -> Target:
        pass

    @abstractmethod
    def decode(self, target: Target) -> Source:
        pass

    def __str__(self):
        return self.__class__.__name__

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

class ListEncoder(Encoder[List[Source], List[Target]]):

    def __init__(self, encoders: List[Encoder[Source, Target]]):
        self.encoders = encoders

    def encode(self, sources: List[Source]) -> List[Target]:
        if len(sources) != len(self.encoders):
            raise Exception('should have the same number of encoders and sources')
        return [encoder.encode(source) for encoder, source in zip(self.encoders, sources)]

    def decode(self, targets: List[Target]) -> List[Source]:
        if len(targets) != len(self.encoders):
            raise Exception('should have the same number of encoders and targets')
        return [encoder.decode(target) for encoder, target in zip(self.encoders, targets)]

class BlockEncoder(Encoder[List[List[Source]], List[List[Target]]]):

    def __init__(self, encoder: Encoder[Source, Target]):
        self.encoder = encoder

    def __str__(self):
        return f'BlockEncoder({self.encoder})'

    def encode(self, blocks: List[List[Source]]) -> List[List[Target]]:
        return [
            [self.encoder.encode(block) for block in row] for row in blocks
        ]

    def decode(self, blocks: List[List[Target]]) -> List[List[Source]]:
        return [
            [self.encoder.decode(block) for block in row] for row in blocks
        ]

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

class DivisibleSizeEncoder(Encoder[np.ndarray, np.ndarray]):

    def __init__(self, width: int, height: int, size: int):
        self.size = size
        self.width = width
        self.height = height
        self.new_w = int(np.ceil(self.width/self.size)*self.size)
        self.new_h = int(np.ceil(self.height/self.size)*self.size)

    def encode(self, rgb_arr: np.ndarray) -> np.ndarray:
        # Pad rows
        if self.new_h > self.height:
            row_pad = np.tile(rgb_arr[-1:, :, :], (self.new_h - self.height, 1, 1))
            rgb_arr = np.vstack((rgb_arr, row_pad))

        # Pad columns
        if self.new_w > self.width:
            col_pad = np.tile(rgb_arr[:, -1:, :], (1, self.new_w - self.width, 1))
            rgb_arr = np.hstack((rgb_arr, col_pad))

        return rgb_arr

    def decode(self, rgb_arr: np.ndarray) -> np.ndarray:
        return rgb_arr[:self.height, :self.width, :]

class ScaleEncoder(Encoder[np.ndarray, np.ndarray]):
    def __init__(self, matrix_scale: np.ndarray):
        self.matrix_scale = matrix_scale

    def encode(self, arr: np.ndarray) -> np.ndarray:
        return np.round(arr / self.matrix_scale).astype(dtype=int) # not //

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.matrix_scale

# <editor-fold desc=" ------------------------ RGB <-> YCbCr ------------------------">

YCbCr = Tuple[np.ndarray, np.ndarray, np.ndarray]

class LinearTransform(Encoder[np.ndarray, np.ndarray]):
    """
    The map    v ->   Av+b
    where A is n x n and b is of dimension n
    """

    def __init__(self, matrix: np.ndarray, offset: np.ndarray):
        self.matrix = matrix
        self.inv_matrix = np.linalg.inv(matrix)
        self.offset = offset

    def encode(self, arr: np.ndarray) -> np.ndarray:
        return np.dot(arr, self.matrix.T) + self.offset

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return np.dot(arr, self.matrix.T- self.offset)

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
        # Perform matrix multiplication and add offset
        ycbcr_arr = np.dot(rgb_arr, self.to_ycbcr_mat.T) + self.to_ycbcr_offset

        # Clip values to the valid range [0, 255] for images
        # ycbcr_arr = np.clip(ycbcr_arr, 0, 255).astype(np.uint8)

        return ycbcr_arr[..., 0], ycbcr_arr[..., 1], ycbcr_arr[..., 2]

    def decode(self, ycbcr: YCbCr) -> np.ndarray:
        ycbcr_arr = np.dstack((ycbcr[0], ycbcr[1], ycbcr[2]))
        ycbcr_arr = ycbcr_arr.astype(np.float32)
        ycbcr_arr -= self.to_rgb_offset
        rgb_arr = np.dot(ycbcr_arr, self.to_rgb_mat.T)
        rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)

        return rgb_arr

def rgb_to_ycbcr(rgb_arr: np.ndarray) -> np.ndarray:
    # Transformation matrix for YCbCr , using BT.601
    # Note that:
    #     Y  = 0.299*R + 0.587*G + 0.114*B
    #     Cb = 0.564*(B-Y)
    #     Cr = 0.713*(R-Y)
    transformation_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.4187, -0.0813]
    ])

    # Offset for Cb and Cr channels (128 for digital YCbCr)
    offset = np.array([0, 128, 128])

    # Perform matrix multiplication and add offset
    ycbcr = np.dot(rgb_arr, transformation_matrix.T) + offset

    # Clip values to the valid range [0, 255] for images
    ycbcr = np.clip(ycbcr, 0, 255).astype(np.uint8)

    return ycbcr

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    # YCbCr to RGB transformation matrix
    transformation_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])

    # Offset for YCbCr
    offset = np.array([0, 128, 128])

    # Ensure the input is float32 for precise calculations
    ycbcr = ycbcr.astype(np.float32)

    # Subtract the offset
    ycbcr -= offset

    # Apply the inverse transformation
    rgb = np.dot(ycbcr, transformation_matrix.T)

    # Clip the values to [0, 255] and convert to uint8
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


# </editor-fold>

# <editor-fold desc=" ------------------------ Space Average ------------------------">

class AverageEncoder(Encoder[np.ndarray, np.ndarray]):

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


def average_block_encode(arr: np.ndarray, block_size: int):
    """
    Averages every block_size x block_size region in a 2D array.

    Args:
        arr (numpy.ndarray): Input 2D array of shape (h, w).
        block_size (int): Size of the block to average.

    Returns:
        numpy.ndarray: Output 2D array of shape (h//block_size, w//block_size).
    """
    h, w = arr.shape
    # Ensure the array dimensions are divisible by block_size
    assert h % block_size == 0 and w % block_size == 0, \
        "Array dimensions must be divisible by block_size"

    # Reshape and average
    new_h = h // block_size
    new_w = w // block_size
    reshaped = arr.reshape(new_h, block_size, new_w, block_size)
    averaged = np.round(reshaped.mean(axis=(1, 3)))

    return averaged


def average_block_decode(arr: np.ndarray, block_size: int):
    return np.repeat(np.repeat(arr, block_size, axis=0), block_size, axis=1)


# </editor-fold>

# <editor-fold desc=" ------------------------ Discrete Cosine Transform ------------------------">

class DCTEncoder(Encoder[np.ndarray, np.ndarray]):

    def encode(self, block: np.ndarray) -> np.ndarray:
        """Compute the 2D Discrete Cosine Transform (DCT-II)."""
        return fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    def decode(self, block: np.ndarray) -> np.ndarray:
        """Compute the 2D Inverse Discrete Cosine Transform (IDCT-II)."""
        return fftpack.idct(fftpack.idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def dct2(block):
    """Compute the 2D Discrete Cosine Transform (DCT-II)."""
    return fftpack.dct(fftpack.dct(block,
                                   axis=0, norm='ortho'),
                       axis=1, norm='ortho')


def idct2(block):
    """Compute the 2D Inverse Discrete Cosine Transform (IDCT-II)."""
    return fftpack.idct(fftpack.idct(block, axis=0, norm='ortho',type=2), axis=1, norm='ortho',type=2)


# </editor-fold>

# <editor-fold desc=" ------------------------ Zig Zag scan ------------------------">

class ZigZagEncoder(Encoder[np.ndarray, np.ndarray]):

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

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        sequence = np.zeros(shape=(self.block_size * self.block_size), dtype=matrix.dtype)

        for index, (row, col) in enumerate(zig_zag_indices(self.block_size)):
            sequence[index] = matrix[row, col]
        return sequence

    def decode(self, sequence: np.ndarray) -> np.ndarray:
        matrix = np.zeros(shape=(self.block_size, self.block_size), dtype=sequence.dtype)

        for index, (row, col) in enumerate(zig_zag_indices(self.block_size)):
            matrix[row, col] = sequence[index]
        return matrix

def zig_zag_indices(size: int):
    even = False
    for diag_len in range(1, size + 1):
        indices = range(diag_len) if even else range(diag_len - 1, -1, -1)
        for i in indices:
            yield i, diag_len - i - 1
        even = not even

    for diag_len in range(size + 1, 2 * size):
        indices = range(2 * size - diag_len) if even else range(2 * size - diag_len - 1, -1, -1)
        for i in indices:
            yield diag_len - size + i, size - i - 1


def zig_zag_block_to_sequence(matrix: np.ndarray, size: int):
    sequence = np.zeros(shape=(size * size), dtype=matrix.dtype)

    for index, (row, col) in enumerate(zig_zag_indices(size)):
        sequence[index] = matrix[row, col]
    return sequence


def zig_zag_sequence_to_block(sequence: np.ndarray, size: int):
    matrix = np.zeros(shape=(size, size), dtype=sequence.dtype)

    for index, (row, col) in enumerate(zig_zag_indices(size)):
        matrix[row, col] = sequence[index]
    return matrix


# </editor-fold>

# <editor-fold desc=" ------------------------ Description ------------------------">

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

def initial_coef_encoding(blocks:List[List[List[int]]]):
    rows = len(blocks)
    cols = len(blocks[0])
    for row in range(rows-1,0,-1):
        for col in range(cols-1,0,-1):
            blocks[row][col][0] -= int( (blocks[row-1][col][0]+blocks[row-1][col-1][0]+blocks[row][col-1][0])//3 )

    for row in range(rows - 1, 0, -1):
        blocks[row][0][0] -= blocks[row - 1][0][0]

    for col in range(cols - 1, 0, -1):
        blocks[0][col][0] -= blocks[0][col - 1][0]

def initial_coef_decoding(blocks:List[List[List[int]]]):
    rows = len(blocks)
    cols = len(blocks[0])
    for row in range(1, rows):
        blocks[row][0][0] += blocks[row - 1][0][0]

    for col in range(1, cols):
        blocks[0][col][0] += blocks[0][col - 1][0]

    for row in range(1, rows):
        for col in range(1, cols):
            blocks[row][col][0] += int( (blocks[row-1][col][0]+blocks[row-1][col-1][0]+blocks[row][col-1][0])//3 )


# </editor-fold>


# <editor-fold desc=" ------------------------ Run Length encoding ------------------------">

ZeroLength_Number = Tuple[int, int]

class RunLengthEncoder(Encoder[List[int], List[ZeroLength_Number]]):

    def __init__(self, list_length: int):
        self.list_length = list_length

    def encode(self, numbers: np.ndarray) -> List[ZeroLength_Number]:
        # Doesn't remember zeros at the end
        numbers = numbers.astype(dtype=int)
        elements = []
        zero_count = 0
        for number in numbers:
            if number == 0:
                zero_count += 1
            else:
                elements.append((zero_count, number))
                zero_count = 0
        return elements

    def decode(self, coded_pairs: List[ZeroLength_Number]) -> np.ndarray:
        elements = np.zeros(self.list_length)
        index = 0
        for zero_count, number in coded_pairs:
            index += zero_count
            elements[index] = number
            index += 1

        return elements

def run_length_encode(numbers: List[int]) -> List[Tuple[int, int]]:
    # Doesn't remember zeros at the end
    elements = []
    zero_count = 0
    for number in numbers:
        if number == 0:
            zero_count += 1
        else:
            elements.append((zero_count, number))
            zero_count = 0
    return elements

def run_length_decode(coded_pairs: List[Tuple[int, int]], list_length: int) -> np.ndarray:
    elements = np.zeros(list_length)
    index = 0
    for zero_count, number in coded_pairs:
        index += zero_count
        elements[index] = number
        index += 1
        # elements += [0] * zero_count
        # elements.append(number)
    # elements += [0] * (list_length - len(elements))
    return elements

# </editor-fold>

def compute_frequencies(blocks):
    counters = defaultdict(lambda: 0)
    total = 0
    for row in blocks:
        for block in row:
            total += len(block)
            for bin_zero_length, bin_number_bits, bin_number in block:
                counters[f'{bin_zero_length}{bin_number_bits}'] += 1
    freq = {symbol: value/total for symbol, value in counters.items()}
    return freq

# <editor-fold desc=" ------------------------ Binary Run Length encoding ------------------------">

Bin_ZeroLen_NumBits_Num = Tuple[str, str, str]

class RL_ToBinaryEncoder(Encoder[List[ZeroLength_Number], List[Bin_ZeroLen_NumBits_Num]]):

    def encode(self, rl_numbers: List[ZeroLength_Number]) -> List[Bin_ZeroLen_NumBits_Num]:
        elements = []
        for zero_length, next_number in rl_numbers:
            while zero_length >= 16:
                elements.append(('1111','0000',''))
                zero_length -= 16

            bin_zero_length = bin(zero_length)[2:].zfill(4)
            bin_number = ('0' if next_number<0 else '') + bin(abs(next_number))[2:]
            bin_number_bits = bin(len(bin_number))[2:].zfill(4)
            elements.append((bin_zero_length, bin_number_bits, bin_number))
        elements.append(('0000','0000',''))
        return elements

    def decode(self, rl_binary: List[Bin_ZeroLen_NumBits_Num]) -> List[ZeroLength_Number]:
        elements = []
        zero_length = 0
        # Don't need to read last element corresponding to end of sequence
        for bin_zero_length, bin_number_bits, bin_number in rl_binary[:-1]:
            cur_zero_length = int(bin_zero_length, 2)
            zero_length += cur_zero_length

            if cur_zero_length == 15 and bin_number == '':
                zero_length += 1
                continue

            sign = 1
            if bin_number[0] == '0':
                sign = -1
                bin_number = bin_number[1:]
            elements.append((zero_length, sign * int(bin_number, 2)))
            zero_length = 0

        return elements

class RL_ToFullBinaryEncoder(Encoder[List[List[List[Bin_ZeroLen_NumBits_Num]]], str]):

    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def encode(self, blocks_data: List[List[List[Bin_ZeroLen_NumBits_Num]]]) -> str:
        return ''.join([''.join([join_binary(block) for block in row]) for row in blocks_data])

    @staticmethod
    def separate_binary(binary_str: str, index):
        data = []
        zero_length_bin = binary_str[index: index + 4]
        bit_length_bin = binary_str[index + 4: index + 8]
        if bit_length_bin == '':
            te = 5
        index += 8
        while zero_length_bin != '0000' or bit_length_bin != '0000':
            bit_length = int(bit_length_bin, 2)
            number_bin = binary_str[index: index + bit_length]
            data.append((zero_length_bin, bit_length_bin, number_bin))
            index += bit_length

            zero_length_bin = binary_str[index: index + 4]
            bit_length_bin = binary_str[index + 4: index + 8]
            index += 8

        data.append((zero_length_bin, bit_length_bin, ''))
        return data, index

    def decode(self, binary_str: str) -> List[List[List[Bin_ZeroLen_NumBits_Num]]]:
        index = 0
        rows = []
        for _ in range(self.num_rows):
            row = []
            for _ in range(self.num_cols):
                block, index = self.separate_binary(binary_str, index)
                row.append(block)
            rows.append(row)
        return rows

def run_length_to_binary(rl_numbers: List[Tuple[int,int]]) -> List[Tuple[str, str, str]]:
    elements = []
    for zero_length, next_number in rl_numbers:
        while zero_length >= 16:
            elements.append(('1111','0000',''))
            zero_length -= 16

        bin_zero_length = bin(zero_length)[2:].zfill(4)
        bin_number = ('0' if next_number<0 else '') + bin(abs(next_number))[2:]
        bin_number_bits = bin(len(bin_number))[2:].zfill(4)
        elements.append((bin_zero_length, bin_number_bits, bin_number))
    elements.append(('0000','0000',''))
    return elements

def join_binary(data: List[Tuple[str, str, str]]):
    return ''.join([f'{zero_length}{bit_length}{number}' for zero_length, bit_length, number in data])

def join_binary_blocks(blocks_data: List[List[List[Tuple[str, str, str]]]]):
    return ''.join([''.join([join_binary(block) for block in row]) for row in blocks_data])

def run_length_from_binary(rl_binary: List[Tuple[str, str, str]]) -> List[Tuple[int,int]]:
    elements = []
    zero_length = 0
    # Don't need to read last element corresponding to end of sequence
    for bin_zero_length, bin_number_bits, bin_number in rl_binary[:-1]:
        cur_zero_length = int(bin_zero_length, 2)
        zero_length += cur_zero_length

        if cur_zero_length == 15 and bin_number == '':
            zero_length += 1
            continue

        sign = 1
        if bin_number[0] == '0':
            sign = -1
            bin_number = bin_number[1:]
        elements.append((zero_length, sign * int(bin_number, 2)))
        zero_length = 0

    return elements

def separate_binary(binary_str: str, index):
    data = []
    zero_length_bin = binary_str[index: index+4]
    bit_length_bin = binary_str[index+4: index+8]
    index += 8
    while zero_length_bin!='0000' or bit_length_bin!='0000':
        bit_length = int(bit_length_bin,2)
        number_bin = binary_str[index: index+bit_length]
        data.append( (zero_length_bin, bit_length_bin, number_bin) )
        index += bit_length

        zero_length_bin = binary_str[index: index+4]
        bit_length_bin = binary_str[index+4: index+8]
        index += 8

    return data, index

def separate_binary_blocks(binary_str:str, num_cols: int, num_rows: int):
    index = 0
    rows = []
    for _ in range(num_rows):
        row = []
        for _ in range(num_cols):
            block, index = separate_binary(binary_str, index)
            row.append(block)
        rows.append(row)
    return rows

# </editor-fold>

# elem_example = [80, 3, -5, 0, 7, 9, 0, 0, -3, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0]
# n = len(elem_example)
# print(n)
# print(elem_example)
# encoded_example = run_length_to_binary(run_length_encode(elem_example))
# print(encoded_example)
# bin_msg = ''.join([''.join(binaries for binaries in trio) for trio in encoded_example])
# print(bin_msg)
# print(len(bin_msg))
# decoded_example = run_length_decode(run_length_from_binary(encoded_example), n)
# print(decoded_example)
# print(elem_example == decoded_example)
