# 4.4.4 多项式回归
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

max_degree = 20  # 多项式最高项数
n_train, n_test = 100, 100  # 训练和测试数据样本数
true_w = np.zeros(max_degree)  # 分配大量空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 构造一个最高项数是3的多项式
# 生成自变量
'''
生成服从正态分布的随机数
numpy.random.normal(loc=0.0, scale=1.0, size=None)
loc: 均值，即分布的中心位置。默认为 0.0
scale: 标准差，即分布的宽度或离散程度。必须 ≥ 0。默认为 1.0
size: 输出的数组形状。
  若为 None（默认），返回一个标量浮点数；
  若为整数 n，返回长度为 n 的一维数组；
  若为元组如 (m, n)，返回 m×n 的二维数组，依此类推。
'''
features = np.random.normal(size=(n_train + n_test, 1))

'''
random.shuffle是就地打乱（in-place shuffle）数组元素顺序的函数
对输入的一维或多维数组沿第一个轴（axis=0）随机打乱行的顺序，不改变数组形状，仅改变元素排列。
'''
np.random.shuffle(features)

'''
对数组元素进行幂运算（element-wise exponentiation） 的函数，支持标量、数组作为底数和指数，并遵循广播（broadcasting）规则。
numpy.power(x1, x2, /, out=None, *, where=True, casting='same_kind', 
            order='K', dtype=None, subok=True[, signature, extobj])
x1：底数（array_like）
x2：指数（array_like）
out：可选，用于指定输出数组（避免内存分配）
where：布尔数组，指定在哪些位置执行计算（其他位置保留 out 原值）
'''
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))

for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, poly_features, labels]
]


def evaluate_loss(net, data_iter, loss):
    """评估模型net在数据集data_iter上的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    # 获取数组最后一个维度的大小
    input_shape = train_features.shape[-1]
    '''
    这行代码定义了一个简单的线性回归模型，用于多项式回归任务：
    模型结构: 创建一个单层的线性网络，将高维多项式特征映射到一维输出
    参数学习: 通过学习权重参数来拟合多项式系数
    无偏置设计: 由于多项式回归通常不需要额外的偏置项，所以设置 bias=False
    '''
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features,
                                 train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = d2l.load_array((test_features,
                                test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch',
                            ylabel='loss',
                            xlim=[1, num_epochs],
                            ylim=[0.001, 100],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (
                evaluate_loss(net, train_iter, loss),
                evaluate_loss(net, test_iter, loss)
            ))
    print('weight:', net[0].weight.data.numpy())


# 三阶多项式函数拟合(正常)
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# 线性函数拟合：欠拟合
# 从多项式特征中选择前2个维度，即1和x
# train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# 高阶多项式函数拟合(过拟合)
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)


