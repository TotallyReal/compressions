import numpy as np

from codecs_funcs.encoding_functions import Source, Target
from codecs_funcs.encoding_functions import Encoder

# in the following, the source and target should be 2d arrays

class Template(Encoder[np.ndarray, np.ndarray]):

    def encode(self, arr: np.ndarray) -> np.ndarray:
        pass

    def decode(self, target: np.ndarray) -> np.ndarray:
        pass

class LeftDiff(Encoder[np.ndarray, np.ndarray]):
    """
    In PNG is called the SUB filter
    """

    def encode(self, arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        result[:, 1:] -= arr[:, :-1]
        return result

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return np.cumsum(arr, axis=1, dtype=arr.dtype)

class UpDiff(Encoder[np.ndarray, np.ndarray]):
    """
    In PNG is called the UP filter
    """

    def encode(self, arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        result[1:, :] -= arr[:-1, :]
        return result

    def decode(self, arr: np.ndarray) -> np.ndarray:
        return np.cumsum(arr, axis=0, dtype=arr.dtype)

class UpLeftDiff(Encoder[np.ndarray, np.ndarray]):
    """
    In PNG is called the AVG filter
    """

    def encode(self, arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        result[1:, 1:] -= (arr[:-1, 1:] + arr[1:, :-1])//2
        return result

    def decode(self, arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        for row in range(1, len(result)):
            for col in range(1, len(result[0])):
                result[row][col] += (result[row-1][col]+result[row][col-1])//2
        return result

class PaethDiff(Encoder[np.ndarray, np.ndarray]):
    """
    In PNG is called the PAETH filter
    """

    def encode(self, arr: np.ndarray) -> np.ndarray:
        print('\nencoding')
        result = arr.copy()
        for row in range(len(result)-1, 0, -1):
            for col in range(len(result[0])-1, 0, -1):
                lv = result[row-1][col]
                uv = result[row][col-1]
                ulv = result[row-1][col-1]
                print(f'{lv=}, {uv=}, {ulv=}')

                if lv < ulv and uv < ulv:
                    result[row][col] -= ulv
                    continue
                if lv < uv:
                    result[row][col] -= uv
                else:
                    result[row][col] -= lv
            print('---')
        print('\n\n')
        return result

    def decode(self, arr: np.ndarray) -> np.ndarray:
        print('decoding')
        result = arr.copy()
        for row in range(1, len(result)):
            for col in range(1, len(result[0])):
                lv = result[row-1][col]
                uv = result[row][col-1]
                ulv = result[row-1][col-1]
                print(f'{lv=}, {uv=}, {ulv=}')

                if lv < ulv and uv < ulv:
                    result[row][col] += ulv
                    continue
                if lv < uv:
                    result[row][col] += uv
                else:
                    result[row][col] += lv
            print('---')
        print('\n\n')
        return result