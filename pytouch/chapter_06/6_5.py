import torch
from torch import nn
from d2l import torch as d2l

'''
6.5.1. 最大汇聚层和平均汇聚层
'''
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

'''
6.5.2. 填充和步幅
'''

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
'''
nn.MaxPool2d 是 PyTorch 中用于执行 二维最大池化（Max Pooling） 操作的神经网络模块，
广泛应用于卷积神经网络（CNN）中，主要用于降低特征图的空间维度、减少计算量、增强平移不变性，并保留最显著的特征。
函数作用：
    对输入的四维张量（通常是卷积后的特征图）在局部感受野内取最大值，从而实现下采样（downsampling）。
    ✅ 输入形状：(N, C, H_in, W_in)
    ✅ 输出形状：(N, C, H_out, W_out)
    通道数 C 不变
    空间尺寸 H, W 通常减小
函数签名：
    torch.nn.MaxPool2d(
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False
    )
参数详解：
    kernel_size：池化窗口大小，如 2 或 (2, 3)
    stride：滑动步长；若未指定，默认等于 kernel_size
    padding：在输入边界填充零的像素数
    dilation：池化窗口内部的膨胀率（注意：PyTorch 的 MaxPool2d 实际不支持 dilation > 1，该参数存在但无效）
    return_indices：是否返回最大值的位置索引（用于 MaxUnpool2d 反池化）
    ceil_mode：尺寸计算时是否向上取整（影响输出大小）
'''
pool2d = nn.MaxPool2d(3)
print(pool2d(X))
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))



