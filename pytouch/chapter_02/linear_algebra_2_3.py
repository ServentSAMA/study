"""
线性代数
"""
import torch

print('张量运算')
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y)

x = torch.arange(4)
print(x)

print(x[2])



# 对称矩阵是矩阵的一种特殊类型
print('对称矩阵')
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])

print(B)

print(B == B.T)

A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B

# 转置矩阵
print('转置矩阵')
print(A.T)

# 降维
print('降维')
x = torch.arange(4, dtype=torch.float32)
x, x.sum()

print(A.shape,x.shape,torch.mv(A,x))

B = torch.ones(4,3)
print('矩阵乘法')
print(A.shape,B.shape)
print(torch.mm(A,B))

# 范数
print('范数')
u = torch.tensor([3.0,-4.0])

print(torch.norm(u))

print(torch.abs(u).sum())

# 张量按照axis求和





