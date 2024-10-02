import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

# 结构信息
print("结构信息")
print(a.shape)
# 维度信息
print("维度信息")
print(a.ndim)

a = np.arange(9).reshape((3, 3))

print(a)
# 统计axis轴为1的值
print(a.sum(axis=1))
print(a.sum(axis=0))

b = np.array([a, a])
print(b)
