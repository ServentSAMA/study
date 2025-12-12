"""
np的shape属性
"""

import numpy as np


def numpy_1_6():
    a = np.array([1, 2, 3, 4, 5, 6])

    print(a.shape)

    print(a.reshape(2, 3))
