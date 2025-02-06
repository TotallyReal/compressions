# https://qoiformat.org/
import cProfile as profile
import pstats
from line_profiler import LineProfiler

pr = profile.Profile()
pr.disable()

from encoding_functions import Encoder, Source, Target
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

class QOIEncoder(Encoder[np.array, List[Chunk]]):

    def encode(self, rgb_seq: np.ndarray) -> List[Chunk]:
        previous = (0,0,0,255)
        a = 255

        previous_pixels = [0] * 64
        chunks = []

        bin_lookup_2 = [bin(i)[2:].zfill(6) for i in range(2**2)]
        bin_lookup_4 = [bin(i)[2:].zfill(6) for i in range(2**4)]
        bin_lookup_6 = [bin(i)[2:].zfill(6) for i in range(2**6)]
        bin_lookup_8 = [bin(i)[2:].zfill(6) for i in range(2**8)]

        run_length = 0
        for r,g,b in rgb_seq:
            r = int(r)
            g = int(g)
            b = int(b)
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

    def decode(self, chunks: List[Chunk]) -> np.array:
        previous = (0,0,0,255)
        r = 0
        g = 0
        b = 0
        a = 255

        previous_pixels = [0] * 64
        sequence = []

        for chunk in chunks:

            # validation = [image_arr[0][i] == tuple(sequence[i][:3]) for i in range(len(sequence))]
            # if len(validation)==15:
            #     temp = 5

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

        return sequence

from PIL import Image
import os
import matplotlib.pyplot as plt


url = '../alice with cards.jpg'
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
rgb_img = np.array(img) #[0:2,:,:]
h, w, _ = rgb_img.shape
print(f'Total size should be: {(h*w*3)//1024} kb')

# ------------------------ compression ------------------------


encoder = QOIEncoder()
image_seq = image_arr.reshape(-1,3).astype(int)
lp = LineProfiler()
# lp.add_function(encoder.encode)
# lp.enable()
pr.enable()
coded = encoder.encode(image_seq)
decompressed_image = encoder.decode(coded)
pr.disable()
# lp.disable()
decompressed_image = np.array(decompressed_image)
decompressed_image = decompressed_image.reshape(h, w, 4)

print_restriction=[]
pstats.Stats(pr).strip_dirs().sort_stats("cumtime").print_stats(*print_restriction)
# lp.print_stats()

bit_str = ''.join([''.join([part for part in elem]) for elem in coded])
print(f'Total bits = {len(bit_str)}')
print(f'in kb = {len(bit_str)//(8*1024)}')


plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
# plt.imshow(rgb_img[230:300, 830:900, :])
plt.imshow(rgb_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(decompressed_image)
plt.axis('off')

plt.show()

