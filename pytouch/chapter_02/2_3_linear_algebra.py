"""
线性代数
"""
import torch


x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x ** y)

# 2.3.2 向量
x = torch.arange(4)
print(x)
print(x[2])

# 2.3.3 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
# 对称矩阵是矩阵的一种特殊类型
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

# 2.3.4 张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 2.3.5 张量算法的基本性质
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, B)
print(A * B) # 将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2,3,4)
print(a + X, (a * X).shape)

# 2.3.6 降维
print('降维')
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
print(A.shape, A.sum())
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





