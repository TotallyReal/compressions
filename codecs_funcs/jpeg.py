from collections import defaultdict
from typing import List

import numpy as np

from codecs_funcs.encoding_functions import (
    Blocker, List2DEncoder, DCTEncoder, ScaleEncoder, ZigZagEncoder, InitialCoefEncoder,
    Encoder, DivisibleSizeEncoder, YCbCrEncoder, ImageFileHeaderEncoder,
    HeaderInfo, Flatten, ListCombiner, BitStreamListEncoder, AverageEncoder, ListEncoder )
from codecs_funcs.run_length import RunLengthEncoder, Bin_ZeroLen_NumBits_Num, RL_ToBinaryEncoder, RL_ToBitStream


class JPEGEncoder(Encoder[np.ndarray, str]):

    def __init__(self):

        self.header = ImageFileHeaderEncoder()

        quant_luma8 = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

        quant_chrome8 = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ])

        self.div16_encoder = DivisibleSizeEncoder(size=16)  # make sure that both height and width are divisible by 16
        self.ycbcr_encoder = YCbCrEncoder()                 # Break to Luma and Chrominance channels

        self.luma_encoders = [
            # Start by blockifying the data     (n/8) x (m/8) x 8 x 8
            Blocker(8),

            # Per block do the following:
            List2DEncoder(DCTEncoder()),                #   1. pixel => DCT coefficients
            List2DEncoder(ScaleEncoder(quant_luma8)),   #   2. Quantization
            List2DEncoder(ZigZagEncoder(8)),            #   3. 8 x 8 => 64

            # Use the block positions in the full image for simple prediction
            InitialCoefEncoder(),                       #   => (n/8) x (m/8) of sequences of (DCT) coefficients

            # No longer need the blocks as part of the full image
            Flatten(),                                  #   => (list of seq of coef, n/8, m/8)

            # Apply encoding to the sequences separately (which appear in the first position)
            ListEncoder(RunLengthEncoder(8*8)).at_position(0),    #  => (ZeroRun, NextNumber) sequences
            ListEncoder(RL_ToBinaryEncoder()).at_position(0),     #  => (ZeroRun, NumBits, NextNumber) sequences (all in binary)
        ]

        # Chrome encoders are the same, but:
        # 1. We start with averaging 2x2 blocks
        self.chroma_encoders = [AverageEncoder(2)] + self.luma_encoders
        # 2. We scale by a different matrix the DCT coefficients
        self.chroma_encoders[3] = List2DEncoder(ScaleEncoder(quant_chrome8))

        # Before we join the chunks into a bit stream, we combine back the 3 color channels
        # according to 4 luma, 1 cb 1 cr blocks
        self.combiner = ListCombiner([4,1,1])

        # Finally, combine everything into a bit_stream
        self.rl_binary_encoder = BitStreamListEncoder(RL_ToBitStream())


    def encode(self, rgb_arr: np.ndarray) -> str:
        h, w, _ = rgb_arr.shape
        header_bit = self.header.encode(HeaderInfo(name='jpeg',width=w, height=h))

        rgb_arr, _, _ = self.div16_encoder.encode(rgb_arr)
        luma, cb, cr = self.ycbcr_encoder.encode(rgb_arr)

        for encoder in self.luma_encoders:
            luma = encoder.encode(luma)

        for encoder in self.chroma_encoders:
            cb = encoder.encode(cb)
            cr = encoder.encode(cr)

        # at index zero we have the data, and at 1, 2 we have the width and height
        data = self.combiner.encode([luma[0], cb[0], cr[0]])
        bit_stream = self.rl_binary_encoder.encode(data)
        return f'{header_bit}{bit_stream}'

    def decode(self, bit_stream: str) -> np.ndarray:
        header_info, index = self.header.decode_bits(bit_stream, 0)
        assert header_info.name == 'jpeg'

        data, _ = self.rl_binary_encoder.decode_bits(bit_stream, index)
        luma, cb, cr = self.combiner.decode(data)

        # add padded width and height parameters
        w = header_info.width
        h = header_info.height
        padded_w = int(np.ceil(w/16)*16)
        padded_h = int(np.ceil(h/16)*16)

        luma = (luma, padded_h//8, padded_w//8)
        cb = (cb, padded_h//16, padded_w//16)
        cr = (cr, padded_h//16, padded_w//16)

        for encoder in reversed(self.luma_encoders):
            luma = encoder.decode(luma)
        for encoder in reversed(self.chroma_encoders):
            cb = encoder.decode(cb)
            cr = encoder.decode(cr)

        decoded = self.ycbcr_encoder.decode((luma, cb, cr))
        decoded = self.div16_encoder.decode((decoded,w,h))

        return decoded


def compute_frequencies(rl_chunks: List[Bin_ZeroLen_NumBits_Num]):
    counters = defaultdict(lambda: 0)
    total = len(rl_chunks)
    for bin_zero_length, bin_number_bits, bin_number in rl_chunks:
        counters[f'{bin_zero_length}{bin_number_bits}'] += 1
    freq = {symbol: value/total for symbol, value in counters.items()}
    return freq
