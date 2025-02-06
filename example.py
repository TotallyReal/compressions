import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from codecs_funcs.encoding_functions import (
    rgb_to_ycbcr, ycbcr_to_rgb,
    average_block_encode, average_block_decode,
    dct2, idct2,
    zig_zag_block_to_sequence, zig_zag_sequence_to_block,
    initial_coef_encoding, initial_coef_decoding,
    run_length_encode, run_length_decode,
    run_length_to_binary, run_length_from_binary,
    join_binary_blocks, separate_binary_blocks,
    compute_frequencies,

    YCbCrEncoder, AverageEncoder)

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


def encode_blocks(func, arr):
    rows = len(arr)
    cols = len(arr[0])
    return [
        [func(arr[i][j]) for j in range(cols)]
        for i in range(rows)
    ]


def apply_to_last_two_dims(func, arr: np.ndarray) -> np.ndarray:
    reshaped = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    processed = np.array([func(block) for block in reshaped])
    return processed.reshape(arr.shape)


def blockify(arr: np.array, block_size: int):
    h, w = arr.shape
    return arr.reshape(
        h // block_size, block_size,
        w // block_size, block_size).swapaxes(1, 2)


def deblockify(arr: np.array, block_size: int):
    h, w, _, _ = arr.shape
    return arr.swapaxes(1, 2).reshape(h * block_size, w * block_size)



def compress_image(
        image_arr_rgb: np.ndarray,
        chrominance_compression_ratio: int = 4,
        block_size: int = 8):
    """
    image_arr_rgb should be the rpg representation of the image.
    Should be of dimension height x width x 3 with entires in [0,255].
    """

    ycbcr_image = rgb_to_ycbcr(image_arr_rgb)

    luma = ycbcr_image[..., 0]
    cb = ycbcr_image[..., 1]
    cr = ycbcr_image[..., 2]

    encoded_luma = luma
    encoded_cb = average_block_encode(cb, chrominance_compression_ratio)
    encoded_cr = average_block_encode(cr, chrominance_compression_ratio)

    encoded_luma = blockify(encoded_luma, block_size)

    encoded_luma = apply_to_last_two_dims(dct2, encoded_luma)
    encoded_luma = apply_to_last_two_dims(lambda block: block // quant_luma8, encoded_luma)
    encoded_luma = encoded_luma.astype(dtype=int)  # ????
    encoded_luma = encode_blocks(lambda block: zig_zag_block_to_sequence(block, 8), encoded_luma)
    initial_coef_encoding(encoded_luma)

    encoded_luma = encode_blocks(run_length_encode, encoded_luma)
    encoded_luma = encode_blocks(run_length_to_binary, encoded_luma)
    freq = compute_frequencies(encoded_luma)
    print(f'total number of symbols is {len(freq)}')
    print(f'number of symbols with freq>0.0001 is {len([fr for fr in freq.values() if fr>0.0001])}')
    for freq_value, symbol in sorted([(fr, symbol) for symbol, fr in freq.items()]):
        print(f'{symbol} : {freq_value}')
    encoded_luma = join_binary_blocks(encoded_luma)

    return encoded_luma, encoded_cb, encoded_cr


def decompress_image(
        encoded_luma, encoded_cb, encoded_cr,
        chrominance_compression_ratio: int = 4,
        block_size: int = 8, image_width=256, image_height=256):

    encoded_luma = separate_binary_blocks(encoded_luma,num_cols=image_width//block_size, num_rows=image_height//block_size)
    encoded_luma = encode_blocks(run_length_from_binary, encoded_luma)
    encoded_luma = encode_blocks(lambda seq: run_length_decode(seq, 64), encoded_luma)

    initial_coef_decoding(encoded_luma)
    encoded_luma = encode_blocks(lambda block: zig_zag_sequence_to_block(block, 8) , encoded_luma)
    encoded_luma = np.array(encoded_luma)
    encoded_luma = apply_to_last_two_dims(lambda block: block * quant_luma8, encoded_luma)
    encoded_luma = apply_to_last_two_dims(idct2, encoded_luma)

    hh, ww, _, _ = encoded_luma.shape
    encoded_luma = deblockify(encoded_luma, block_size)

    decoded_luma = encoded_luma
    decoded_cb = average_block_decode(encoded_cb, chrominance_compression_ratio)
    decoded_cr = average_block_decode(encoded_cr, chrominance_compression_ratio)

    # retrieve image:
    decoded_ycbcr_arr = np.dstack((decoded_luma, decoded_cb, decoded_cr))

    rgb_image = ycbcr_to_rgb(decoded_ycbcr_arr)
    return rgb_image

url = 'alice with cards.jpg'
img = Image.open(url)

image_arr = np.array(img)
# for each pixel we have the 3 RGB values. Taking their averages, gives us a
# gray scale version
image_grayscale = np.round(np.mean(img, -1))

# ------------------------ Image size ------------------------

print('Image size data:')
size_kb = int(os.path.getsize(url))//1024
print(f'File size is {size_kb} kb')

# The image is a 3D array of size width x height x 3, and each entry
# is an uint8, namely 1 byte.
print(f'Image array dimension : {image_arr.shape}')
print(f'Image array type : {image_arr.dtype}')
h, w, _ = np.array(img).shape
print(f'Total size should be: {(h*w*3)//1024} kb')

# ------------------------ compression ------------------------

rgb_img = np.array(img)
image_height, image_width, _ = rgb_img.shape

from codecs_funcs.encoding_functions import (
    Blocker, BlockEncoder, DCTEncoder, ScaleEncoder, ZigZagEncoder, InitialCoefEncoder, RunLengthEncoder,
    RL_ToBinaryEncoder, RL_ToFullBinaryEncoder, DivisibleSizeEncoder, HookEncoder)

chroma_ratio = 2
size_encoder = DivisibleSizeEncoder(width=image_width, height=image_height, size=8*chroma_ratio)
ycbcr_encoder = YCbCrEncoder()
                                                                # n x m matrix
blocker = Blocker(8)                                            # =>  (n/8) x (m/8) x 8 x 8
                                                                # Per 8 x 8 block:
dct_block_encoder    = BlockEncoder(DCTEncoder())               #   1. pixel => DCT coefficients
scale_block_encoder  = BlockEncoder(ScaleEncoder(quant_luma8))  #   2. Quantization
zig_zag_encoder      = BlockEncoder(ZigZagEncoder(8))           #   3. 8 x 8 => 64
initial_coef_encoder = InitialCoefEncoder()                     # Normalize first element in each list
                                                                # Per sequence:
run_len_encoder      = BlockEncoder(RunLengthEncoder(8*8))      #   1. numbers => list[ (zero_count, number) ]
rl_binary_encoder    = BlockEncoder(RL_ToBinaryEncoder())       #   2.         => list[ (zero_count, bit_for_number, number) ]
full_binary          = RL_ToFullBinaryEncoder(                  #   3.         => single binary (in str)
                            num_rows=size_encoder.new_h//8, num_cols=size_encoder.new_w//8)
full_binary2         = HookEncoder(RL_ToFullBinaryEncoder(                  #   3.         => single binary (in str)
                            num_rows=size_encoder.new_h//(8*chroma_ratio), num_cols=size_encoder.new_w//(8*chroma_ratio)))

luma_encoders = [
    blocker,
    dct_block_encoder,
    scale_block_encoder,
    zig_zag_encoder,
    initial_coef_encoder,
    run_len_encoder,
    rl_binary_encoder,
    full_binary
]

chroma_encoders = [
    AverageEncoder(chroma_ratio),
    blocker,
    dct_block_encoder,
    BlockEncoder(ScaleEncoder(quant_chrome8)),
    zig_zag_encoder,
    initial_coef_encoder,
    run_len_encoder,
    rl_binary_encoder,
    full_binary2
]

def compress_image2(source: np.ndarray):
    source = size_encoder.encode(source)
    luma, cb, cr = ycbcr_encoder.encode(source)
    for encoder in luma_encoders:
        luma = encoder.encode(luma)
    for encoder in chroma_encoders:
        cb = encoder.encode(cb)
        cr = encoder.encode(cr)
    return luma, cb, cr

def decompress_image2(target):
    luma, cb, cr = target
    for encoder in reversed(luma_encoders):
        luma = encoder.decode(luma)
    for encoder in reversed(chroma_encoders):
        cb = encoder.decode(cb)
        cr = encoder.decode(cr)
    decoded = ycbcr_encoder.decode((luma, cb, cr))
    return size_encoder.decode(decoded)


compressed_image   = compress_image2(rgb_img)
decompressed_image = decompress_image2(compressed_image)
original_size_bit = image_height*image_width*3*8
print(f'Original size is {original_size_bit//(1024*8)} kb')
compressed_size_bit = sum([len(compressed_image[i]) for i in range(3)])
print(f'After compression is {compressed_size_bit//(1024*8)} kb')
print(f'So we have a {compressed_size_bit / original_size_bit} compression ratio')
print(f'luma to chroma ratio = {len(compressed_image[0])/(len(compressed_image[1])+len(compressed_image[2]))}')

plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
# plt.imshow(rgb_img[230:300, 830:900, :])
plt.imshow(rgb_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(decompressed_image)
plt.axis('off')

plt.show()