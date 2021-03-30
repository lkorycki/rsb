import itertools
from functools import reduce
from typing import Callable

import numpy as np


class CollectionUtils:

    @staticmethod
    def ensure_arr_size(arr: np.ndarray, y: int):
        if len(arr.shape) == 1:
            return np.hstack((arr, np.zeros(y - len(arr) + 1))) if len(arr) - 1 < y else arr
        else:
            return np.hstack((arr, np.zeros((arr.shape[0], y - arr.shape[1] + 1, arr.shape[2])))) if arr.shape[1] - 1 < y else arr

    @staticmethod
    def ensure_list2d_size(lst: list, y: int, element_creator: Callable[[], any]):
        if len(lst[0]) > y:
            return lst

        for l in lst:
            l_len = len(l) - 1
            for _ in range(y - l_len):
                l.append(element_creator())

        return lst

    @staticmethod
    def split_list(lst, chunk_size: int):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    def flatten_list(lst):
        return list(itertools.chain.from_iterable(lst))

