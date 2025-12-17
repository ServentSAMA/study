"""
4.1. 多层感知机
"""
import torch
from d2l import torch as d2l
'''
4.1.2. 激活函数
    激活函数（activation function）通过计算加权和并加上偏置来确定神经元是否应该被激活， 
    它们将输入信号转换为输出的可微运算。 大多数激活函数都是非线性的。
'''
# ReLU函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

# sigmoid函数
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 清除之前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
