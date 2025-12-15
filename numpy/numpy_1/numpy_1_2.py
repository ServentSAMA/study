import numpy as np


def numpy_1_2():
    a = np.array([1, 2, 3])
    print(type(a))
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)
    print(b)
    print(b.T)
    print(a.T)
    print(a.data)
    print(a.dtype)
    print(a.flags)
    print(b.flags)
    print(a.flat[1])
    print(b.flat[4])
