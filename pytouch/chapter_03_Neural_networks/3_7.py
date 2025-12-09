"""softmax回归简洁实现"""
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''
softmax回归的输出层是一个全连接层。因此，为了实现我们的模型，我们只需要在
Sequential中添加一个带有10个输出的全连接层
'''

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
'''
用于多分类任务的标准损失函数，它将 LogSoftmax 和 NLLLoss（负对数似然损失） 合并为一步，数值更稳定、使用更便捷。

计算预测 logits（未归一化的分数）与真实标签之间的交叉熵损失，适用于单标签多分类问题（每个样本只属于一个类别）。
torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean',
    label_smoothing=0.0
)

weight：手动指定每个类别的权重（形状 [C]），用于处理类别不平衡。
reduction： 'none'：返回每个样本的损失
            'mean'：返回 batch 平均损失
            'sum'：返回 batch 总损失
'''
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,trainer)
