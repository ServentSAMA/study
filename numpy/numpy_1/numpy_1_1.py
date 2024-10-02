import numpy as np
import numpy.random as nr


def numpy_1_1_4():
    """
    创建np数组
    """
    # 创建类型为ndarray类型的数组
    a = np.array([1, 2, 3])

    '''
    对np数组乘3，会对数组中所有的值进行乘三
    '''
    print('-----------')
    print('数组运算，np会对数组中的所有元素进行乘3')
    print('-----------')
    print(a * 3)

    print([1, 2, 3] * 3)

    b = np.array([4, 5, 6])
    # 对np数组进行加减乘除
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)

    '''
    np矩阵乘积位哈达玛积的计算
    两个矩阵的哈达玛积是逐个元素的相乘并求和，最终得到的是同形状的新矩阵。
    '''
    print('-----------')
    print('哈达玛积：')
    print('-----------')
    print(np.dot(a, b))
    d = np.array([[1, 2], [4, 5]])
    e = np.array([[7, 8], [10, 11]])
    print(np.dot(d, e))
    # 创建连续数值的np数组
    print('-----------')
    print('连续数组')
    print('-----------')
    print(np.arange(10))
    print('-----------')
    print('将0-10之间划分为15等分')
    print('-----------')
    print(np.linspace(0, 10, 15))

    #  创建二维数组

    c = np.array([[1, 2, 3], [4, 5, 6]])
    print(c)
    print(c.shape)

    '''
    对数组求和
    
    '''
    print('-----------')
    print('对矩阵进行求和')
    print('-----------')
    print(c.sum())
    # 指定坐标轴axis
    print(c.sum(axis=1))
    '''
    对数组的形状进行修改
    '''
    print('-----------')
    print('对数组的形状进行修改')
    print('-----------')
    print(c.reshape(3, 2))
    '''
    [[1 2]
     [3 4]
     [5 6]]
    '''
    print(c.reshape(6, 1))

    '''
    对矩阵进行转置
    矩阵转置：将矩阵的列和行进行交换
    [[1 2 3]
     [4 5 6]]
    
    to
    
    [[1 4]
     [2 5]
     [3 6]]
    '''
    print('-----------')
    print('对矩阵进行转置')
    print('-----------')
    print(c.transpose())
    print(c.T)
    print(np.transpose(c))
    '''
    随机数
    '''
    print('-----------')
    print('创建随机数')
    print('-----------')
    '''
    创建复合标准正态分布的随机值
    '''
    randn = nr.randn
    print(randn)
    '''
    返回0-1的随机数
    '''
    rand = nr.rand
    print(rand)
    '''
    对randn指定形状
    '''
    randn = nr.randn(2, 3)

    print(randn)

    '''
    索引与切片
    '''

    print(a)
    print(a[0])
    # 倒序
    print(a[::-1])
    print(a[0:2])

    a = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
    b = np.arange(3 * 4 * 5 * 6)[::-1].reshape((5, 4, 6, 3))
