import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装无误

'''
2.1.1 数据操作
'''

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
# reshape转换为矩阵
X = x.reshape(3, 4)
print(X)

X = X.reshape(12)
print(X)

# 我们不需要指定每个维度来改变形状，
# 我们如果想要的是高度和宽度的形状，在知道宽度时，
# 高度就可以被自动计算得出
print(x.reshape(-1, 3))

# 正态分布
randn = torch.randn(3, 4)
print(randn)
'''
2.1.2 运算符
'''
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2,2,2,2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y) # 幂运算

# 

print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((X,Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())