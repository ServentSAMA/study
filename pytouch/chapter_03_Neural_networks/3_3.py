import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# features特征，labels标签
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# 使用框架定义好的层
# Sequential可以理解为把层一个一个放在list里
'''
nn.Sequential: 这是一个容器模块，用于按顺序组织多个神经网络层。它会按照传入的顺序依次执行每个子模块。
nn.Linear(input_shape, 1, bias=False): 创建一个线性变换层
2: 输入维度为2
1: 输出维度为1，因为这是一个回归任务，需要预测单个数值
'''
net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
