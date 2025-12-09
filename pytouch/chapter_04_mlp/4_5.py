"""
高维线性回归
下面我们将从头开始实现权重衰减，只需将 L2 的平方惩罚添加到原始目标函数中
"""
import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    """
    初始化模型参数

    返回:
        list: 包含权重矩阵w和偏置b的列表，其中w服从正态分布N(0,1)，b初始化为0，
              两者都设置requires_grad=True以支持梯度计算
    """
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_panelty(w):
    """
    计算权重参数的L2范数惩罚项（平方和的一半）

    参数:
        w (torch.Tensor): 权重参数张量

    返回:
        torch.Tensor: L2范数惩罚值
    """
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    """
    训练线性回归模型，可选择是否使用L2正则化（权重衰减）

    参数:
        lambd (float): 正则化系数，控制L2惩罚项的强度。
                      当lambd=0时，不使用正则化；当lambd>0时，启用权重衰减

    功能说明:
        - 使用合成数据进行训练和测试
        - 实现带L2正则化的线性回归
        - 可视化训练过程中的损失变化
        - 输出最终权重的L2范数
    """
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 在损失函数中加入L2正则化项，实现权重衰减
            l = loss(net(X), y) + lambd * l2_panelty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
            if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1, (
                    d2l.evaluate_loss(net, train_iter, loss),
                    d2l.evaluate_loss(net, test_iter, loss)
                ))
    print('w的L2范数是：', torch.norm(w).item())

def train_concise(wd):
    """
    使用简洁的实现，训练线性回归模型，可选择是否使用L2正则化（权重衰减）
    """
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, "weight_decay": wd},
        {"params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (
                d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)
            ))
    print('w的L2范数是：', net[0].weight.norm().item())


# 忽略正则化直接训练
# 这里训练误差有了减少，但测试误差没有减少， 这意味着出现了严重的过拟合。
# train(lambd=0)
# 我们使用权重衰减来运行代码。 注意，在这里训练误差增大，但测试误差减小。 这正是我们期望从正则化中得到的效果。
# train(lambd=3)

train_concise(wd=3)
