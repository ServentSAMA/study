import numpy as np


def numpy_1_8():
    a = np.random.randint(10, size=100)

    n = a % 3 == 0

    print(a[n])
