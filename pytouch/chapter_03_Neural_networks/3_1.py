import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 3.1.2 向量化加速
'''
在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本
为了实现这一点，需要我们对计算进行向量化，从而利用《线性代数库》
'''


class Timer:  #@save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


n = 10000
# 新建两个1万数量，值为1的tensor
a = torch.ones(n)
b = torch.ones(n)
# 新建一个1万值为0的tensor
c = torch.zeros(n)

timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]

print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')

# 3.1.3 正态分布与平方损失
'''
正态分布和线性回归之间的关系很密切，正态分布也称为高斯分布
'''


def normal(x, mu, sigma):
    """
    :param mu: 均值
    :param sigma: 标准差
    """
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


x = np.arange(-7, 7, 0.01)
print(x)
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std{sigma},' for mu, sigma in params])
