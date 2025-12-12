import torch

# 是否支持cuda
print(torch.cuda.is_available())
# 没有初始化的5*3矩阵
x = torch.empty(5, 3)
print(x)
# 随机初始化矩阵
y = torch.rand(5, 3)

print(y)
# 创建值为0的矩阵
zero_x = torch.zeros(5, 3, dtype=torch.long)

print(zero_x)

tensor1 = torch.tensor([5.5, 3])

print(tensor1)

# 启用requires_grad=True 来追踪该变量相关的计算操作
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2

print(y)

print(y.grad_fn)

z = y * y * 3

out = z.mean()

print('z=',z)
print('out=',out)



