"""
数组的真假值判断函数 
np.all(a, axis=None, out=None, keepdims=False)
参数介绍
a：array_like 类似数组的对象，用于指定输入的数组，或者可以变换成数组的对象

axis：None、int或int元组，初始值为None，用于指定从那个坐标轴上进行访问

out：ndarray，初始值为None，用于指定保存结果的数组

keepdims：bool，初始值为False，用于指定在输出结果时，对于元素数量为1的维度是否也原样保留。
          如果指定为True，针对原有的数组自动使用广播机制进行计算

返回值
"""
import numpy as np

a = np.array([[1, 1, 1],
              [1, 0, 0],
              [1, 0, 1]])

# print(np.all(a, keepdims=True)) # 一个不为一，就返回false


b = np.ones((3, 3))
# print(np.all(b)) # 全为1时才返回true

# print(a < 2) # a中的元素全部小于2时才返回true

# 可以将a<2的元素打印出来，打印的是相同形状的布尔类型数据
# print(np.all(a < 2))

# print(np.all(a, axis=0)) # 从行方向上遍历元素

# print(np.all(a, axis=1)) # 从列方向上遍历元素

a[2, 0] = 0  # 将第三行第一列的元素更改为0

print(np.all(a, axis=0))

"""
np.any()的使用方法和all完全相同，但是返回结果不同，any是数组中有一项满足就会返回True
"""
