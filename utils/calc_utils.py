import math

import torch
from torch.tensor import Tensor
import numpy as np


class CalculationsUtils:

    @staticmethod
    def div(a: float, b: float):
        return a / b if b else 0.0

    @staticmethod
    def div_tensor(t: Tensor, d: float):
        return t / d if d else torch.zeros(len(t))

    @staticmethod
    def normalize(t: Tensor):
        s = t.sum()
        return t / s if s > 0.0 else t

    @staticmethod
    def sum_arrays(arrays):
        out = []
        active = set(range(len(arrays)))

        i = 0
        while active:
            out.append(0.0)
            to_remove = set()

            for a_idx in active:
                if i > len(arrays[a_idx]) - 1:
                    to_remove.add(a_idx)
                else:
                    out[-1] += arrays[a_idx][i]

            active -= to_remove
            i += 1

        return np.array(out[:-1])

    @staticmethod
    def log(x):
        if x > 0.0: return math.log(x)
        return float('-inf') if x == 0.0 else float('NaN')
