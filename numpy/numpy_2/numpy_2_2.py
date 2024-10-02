import numpy as np

a = np.arange(12)

b = np.arange(12).reshape((3, 4))
print(b)
"""
 在np中同样有append函数，但是功能和python的extend接近， 
 再加上内部实现时会对元素数据进行复制，所以实际执行起来的速度比想象中的要慢
"""
"""
# 在数组末尾添加指定的元素并生成新的数组的函数
# 语法：
# np.append(arr, values, axis=None)

append的返回值:
返回添加了元素之后得到的ndarray
不指定axis，无论是什么形式的矩阵都会返回一维数组
"""
print(np.append(b, [12, 13, 14, 15]))
'''
values – 这些值被附加到' arr '的副本上。
它必须具有正确的形状(与' arr '相同的形状，不包括' axis ')。
如果未指定' axis '，则' values '可以是任何形状，并且在使用前将被平展。
'''
print(np.append(b, [[12, 13, 14, 15]], axis=0))

a = np.arange(12)
# 在后面添加新数组
print(np.append(a, [6, 4, 2]))


b = np.arange(12).reshape(3, 4)

print(b)
print(np.append(b, [[12, 13 ,14, 15]], axis=0))
# 指定axis后要添加的数组形式和被添加数组形式不一致就会报错
print(np.append(b, [12, 13, 14, 15], axis=0))


c = np.arange(12).reshape((3, 4))



