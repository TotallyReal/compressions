import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from typing import Any

from codecs_funcs.encoding_functions import Encoder
from codecs_funcs.jpeg import JPEGEncoder
from codecs_funcs.qoi import QOIEncoder


def example_image_compression(
        encoder: Encoder[np.ndarray, Any],
        image_url:str, image_type: str = 'RGB'):

    # ------------------ Load image and print some data ------------------
    img = Image.open(image_url).convert(image_type)
    size_kb = int(os.path.getsize(image_url))//1024
    print(f'Image file size is {size_kb} kb')

    image_arr = np.array(img) #[:15,:20,:]
    print(f'Image array dimension : {image_arr.shape}')
    print(f'Image array type : {image_arr.dtype}')
    h, w, color_channels = image_arr.shape
    total_size = (h*w*color_channels)
    print(f'Total size should be: {total_size//1024} kb')
    if color_channels == 4:
        print(f'Total size without alpha should be: {(h*w*3)//1024} kb')

    # ------------------ Compression ------------------
    bit_stream = encoder.encode(image_arr)
    decompressed_image = encoder.decode(bit_stream)

    if type(bit_stream) == str:
        print(f'Compressed size is: {len(bit_stream) // (8*1024)} kb')
        print(f'Compression ratio is: {len(bit_stream)/(total_size*8)}')


    # ------------------ Show before and after compression ------------------
    plt.figure(figsize=(8, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_arr)
    # plt.imshow(rgb_img[230:300, 830:900, :])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_image)
    plt.axis('off')

    plt.show()


# example_image_compression(QOIEncoder(), 'images/alice with cards.jpg', 'RGBA')
example_image_compression(JPEGEncoder(), 'images/alice with cards.jpg', 'RGB')

