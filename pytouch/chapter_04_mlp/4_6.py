"""
暂退法
"""
import torch
from torch import nn
from d2l import torch as d2l

# 从零开始实现

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况下，所有元素都会丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 所有元素都会保留
    if dropout == 0:
        return X
    '''
        torch.rand(*size, out=None, dtype=None, layout=torch.strided, 
           device=None, requires_grad=False)
           *size：int 或 tuple of ints 输出张量的形状。可传多个整数或者元组
           out：Tensor 可选输出张量（必须与指定形状兼容）。
           dtype：torch.dtype
           device：张量所在设备（如 'cuda', 'cuda:0'）。
           requires_grad：是否记录计算图以支持自动求导（常用于模型参数）。
    '''
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

X = torch.arange(16, dtype=torch.float32).reshape((2, 8))

# print(X)
# print(dropout_layer(X, 0))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.5, 0.2

class Net(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hiddens1,
                 num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
'''
定义了一个多层感知机（MLP）神经网络模型，用于图像分类任务（如 MNIST 手写数字识别）
nn.Sequential 是 PyTorch 中的一种容器模块，按顺序依次执行传入的子模块（layers）
数据从输入到输出自动依次流经每个层，无需手动定义 forward 方法。
nn.Flatten()：将输入张量展平为一维向量（保留 batch 维度）。
    典型输入：图像数据，如 (batch_size, 1, 28, 28)（MNIST 单通道 28×28 图像）。
    输出形状：(batch_size, 784)，因为 28×28=784。
    相当于 x.view(x.size(0), -1)。
nn.Linear(784, 256)：
    全连接层（线性变换）
    输入特征数：784（来自展平后的图像）
    输出特征数：256（第一隐藏层的神经元数量）
    参数量：784×256+256=200,960
nn.ReLU()：
    激活函数：ReLU
    引入非线性，使网络能学习复杂模式。
nn.Dropout(dropout1)：
    Dropout 层：在训练时以概率 dropout1 随机将部分神经元输出置零。
    目的：防止过拟合（regularization）。
    dropout1 是一个浮点数（如 0.2 表示 20% 的神经元被丢弃）。
    ⚠️：在推理（eval）模式下自动关闭。
nn.Linear(256, 256)：
    第二个隐藏层，输入/输出均为 256 维（保持维度不变）。
nn.ReLU()
    再次应用 ReLU 激活。
nn.Dropout(dropout2)：
    第二个 Dropout 层，丢弃率由 dropout2 控制（可与 dropout1 不同）。
nn.Linear(256, 10)：
    最后的输出层：
        输入：256 维（来自最后一个隐藏层）
        输出：10 维（对应 10 个类别，如 MNIST 的数字 0~9）
    此层不加 softmax！通常配合 nn.CrossEntropyLoss 使用（该损失函数内部处理 softmax）。
'''
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)


trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.plt.show()
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)