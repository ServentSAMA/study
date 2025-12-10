"""
5.4 自定义层
深度学习成功背后的一个因素是神经网络的灵活性： 我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。
例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。
有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。
在这些情况下，必须构建自定义层。本节将展示如何构建自定义层。
"""

import torch
import torch.nn.functional as F
from torch import nn

'''
不带参数的层
'''


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
# layer(torch.FloatTensor([1, 2, 3, 4, 5]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
# Y.mean()

'''
带参数的层
'''


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)


# linear.weight

# 设计一个返回输入数据的傅立叶系数前半部分的层。
class HalfFFT(nn.Module):
    def __init__(self):
        super(HalfFFT, self).__init__()

    def forward(self, X):
        fft_f = torch.fft.fft(X)
        n = fft_f.shape[-1]
        half_n = n // 2 + 1
        return fft_f[..., :half_n]


myNet2 = HalfFFT()
print(myNet2(torch.rand(2, 3)))
