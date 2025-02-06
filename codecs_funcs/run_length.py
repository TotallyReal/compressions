import numpy as np
from typing import List, Tuple

from codecs_funcs.encoding_functions import Encoder, BitStreamEncoder

"""
The idea behind the Run Length encoding for a list of integers is that if there is a long sequence of zeros,
instead of repeating 0 many times, it is better to save just the length of this run.
Here we do it only for zeros, though in general we can encode runs of other numbers as well.

For clarity, we do the full encoding into a bit stream in several steps:
1. Start with a list of integers.
2. RunLengthEncoder: Transform to a list of pairs: (zero_length, next_number). 
   Each such pair indicates that we zero zero_length zeros follows by next_number which is nonzero.
   If the original list ends with a sequence of zeros, we just don't save it. Instead, when decoding we know
   in advance the length of the original list, so we can add back the zeros.
3. RL_ToBinaryEncoder: Transform to a list of binary triples: (zero_length, num_bit_number, next_number)
   This is done in preparation to combine the sequence into a binary stream
   a. zero_length   : is an exactly four bits number (0..15) indicating the number of zeros
   b. num_bit_number: is an exactly four bits number indicating how many bits we need to represent the next number.
                      This means that we assume (!) that we don't need more that 15 bits to represent it.
   c. next_number   : a binary expansion of the next number. Start the expansion with zero to indicate that the 
                      number is negative.

   If there is a more than 16 length sequence of zeros, we use a (15, 0, '') symbol to represent a subsequence
   of 16 zeros.
   The sequence always ends with (0,0,''). This is used to find the end of the sequence when decoding back
   the bit stream.
4. RL_ToBitStream: Joins all the symbols together into a single bitstream.

Example for a 32 length sequence:
1. [10, 0, 0, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0 ,0 ,0, 0, 0]
2. [(0,10), (2,5), (0,3), (18,-2), (1,-1)]
3. [(0000,0100,1010), (0010,0011,101), (0000,0010,11), (1111,0000,''), (0010,0011,010), (0001,0010,01), (0000,0000,'')]
4. 00000100101000100011101000000101100110011010
"""

ZeroLength_Number = Tuple[int, int]


class RunLengthEncoder(Encoder[np.ndarray, List[ZeroLength_Number]]):

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


Bin_ZeroLen_NumBits_Num = Tuple[str, str, str]


class RL_ToBinaryEncoder(Encoder[List[ZeroLength_Number], List[Bin_ZeroLen_NumBits_Num]]):

    def encode(self, rl_numbers: List[ZeroLength_Number]) -> List[Bin_ZeroLen_NumBits_Num]:
        elements = []
        for zero_length, next_number in rl_numbers:
            while zero_length >= 16:
                elements.append(('1111', '0000', ''))
                zero_length -= 16

            bin_zero_length = bin(zero_length)[2:].zfill(4)
            bin_number = ('0' if next_number < 0 else '') + bin(abs(next_number))[2:]
            bin_number_bits = bin(len(bin_number))[2:].zfill(4)
            elements.append((bin_zero_length, bin_number_bits, bin_number))
        elements.append(('0000', '0000', ''))
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


class RL_ToBitStream(BitStreamEncoder[List[Bin_ZeroLen_NumBits_Num]]):

    def encode(self, data: List[Bin_ZeroLen_NumBits_Num]) -> str:
        return ''.join([f'{zero_length}{bit_length}{number}' for zero_length, bit_length, number in data])

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[List[Bin_ZeroLen_NumBits_Num], int]:
        data = []
        zero_length_bin = bit_stream[index: index + 4]
        bit_length_bin = bit_stream[index + 4: index + 8]

        index += 8
        while zero_length_bin != '0000' or bit_length_bin != '0000':
            bit_length = int(bit_length_bin, 2)
            number_bin = bit_stream[index: index + bit_length]
            data.append((zero_length_bin, bit_length_bin, number_bin))
            index += bit_length

            zero_length_bin = bit_stream[index: index + 4]
            bit_length_bin = bit_stream[index + 4: index + 8]
            index += 8

        data.append((zero_length_bin, bit_length_bin, ''))

        return data, index