import torch

'''
创建torch中的默认张量
'''
tensor = torch.arange(12)
print(tensor)

'''
改变形状 数据预处理
将一维数组转换成二维数组
二维数组大小的乘积需等于一维数组的大小
'''

tensor = torch.reshape(tensor,(3,4))
print(tensor)

# 创建全是1的三位数组
x = torch.ones((2, 3, 4))
print(x)

x = torch.randn((3, 4))
print(x)

# 广播机制
# 矩阵a将复制列， 矩阵b将复制行，然后再按元素相加。
a = torch.arange(3).reshape(3, 1) + 1
b = torch.arange(2).reshape(1, 2) + 1
print(a+b)


