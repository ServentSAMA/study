"""
6.3 填充和步幅
"""
import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])
'''
在 PyTorch 的 nn.Conv2d 中，填充（padding） 和 步幅（stride） 是两个关键超参数，它们直接影响卷积操作的输出尺寸、感受野大小以及特征图的空间分辨率。
步幅
    步幅（stride） 指的是卷积核（filter）在输入特征图上每次滑动的像素步长。
    作用：
        控制下采样程度：stride > 1 会减小输出特征图的尺寸。
        减少计算量和参数量。
        增大有效感受野（因为跳过了部分区域）。
填充：
    填充（padding） 指在输入特征图的四周添加额外的像素（通常为 0），以控制输出尺寸。
    作用：
        防止边界信息丢失：卷积核在边缘时也能有完整的感受野。
        控制输出尺寸：通过适当填充，可使输出与输入尺寸相同（“same padding”）。
'''
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
# comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# comp_conv2d(conv2d, X).shape


