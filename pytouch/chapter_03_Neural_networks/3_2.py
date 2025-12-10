"""
线性回归从零开始实现
"""
import random
import torch
from d2l import torch as d2l

'''
生成数据集
'''


def synthetic_data(w, b, num_examples):
    """
    生成y=Xw+b+噪声

    :param w:
    :param b:
    :param num_examples: 生成的样本
    """
    # 生成正态分布随机数
    # 均值为0(第一个参数)，方差为1(第二个参数)的随机数，第三个参数为大小
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 对X和w求矩阵积，加上偏差b
    '''
        用于执行张量矩阵乘法（matrix multiplication） 的核心函数，支持从向量、矩阵到高维张量的广泛广播和批量运算。
        它是深度学习中最常用的线性代数操作之一。
        输入维度	行为
        1D × 1D	点积（内积） → 标量
        2D × 2D	标准矩阵乘法 → 矩阵
        1D × 2D	向量左乘矩阵（自动升维）→ 向量
        2D × 1D	矩阵右乘向量 → 向量
        ≥3D	    批量矩阵乘法（Batched Matrix Multiplication）
        ✅ 支持广播机制（broadcasting），适用于批量处理（如 (B, M, N) × (B, N, K) → (B, M, K)）。
        
        
    '''
    y = torch.matmul(X, w) + b
    # 相当于上边的公式
    y += torch.normal(0, 0.01, y.shape)
    # X和y做成列向量返回
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

# 生成散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy())


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机样本
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# 定义初始化模型参数
print('定义初始化模型参数')
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    """
        线性回归模型
        matmul()
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
        均方损失函数
        回归问题中最常用的损失函数是平方误差函数
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
        小批量随机梯度下降
        梯度会存在grad里面
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 超参数
lr = 0.03
# 每轮遍历数据集的次数
num_epochs = 3
# 模型
net = linreg
# 损失函数
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 'X'和'y'的小批量损失
        l = loss(net(X, w, b), y)
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        # 使用参数梯度更新
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
