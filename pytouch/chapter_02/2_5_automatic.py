"""
自动微分
"""

import torch

x = torch.arange(4.0)

print(x)
#
# 注意：torch.arange() 默认生成的是 整数类型 张量（如 torch.int64），而 整数类型的张量不能设置 requires_grad=True！
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad) # 默认值是 None

y = 2 * torch.dot(x, x) # 点积x等于14，关于x的梯度为4x
print(y)
print(y.backward())
print(x.grad)
print(x.grad == 4 * x)

# 在默认情况下pytorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 2.5.2 非标量变量的反向传播
# 当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。 对于高阶和高维的y和x，求导的结果可以是一个高阶张量。
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
print("2.5.2 非标量变量的反向传播")
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# 2.5.3 分离计算
print("2.5.3 分离计算")
x.grad.zero_()
print(x)
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)



