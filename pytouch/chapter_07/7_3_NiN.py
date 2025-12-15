"""
7.3. 网络中的网络（NiN）
"""
import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
'''
默认学习率0.1，训练次数10，批量大小：128
loss 0.360, train acc 0.866, test acc 0.862
2681.6 examples/sec on cuda:0
'''
'''
1、调整NiN的超参数，以提高分类准确性。
    只调整学习率为：0.05，在第五次训练时出现欠拟合，训练损失高，AAC低
    loss 0.379, train acc 0.860, test acc 0.875
    2682.9 examples/sec on cuda:0
    只调整训练次数：第四次出现了欠拟合
    loss 0.798, train acc 0.708, test acc 0.707
    2698.1 examples/sec on cuda:0
'''
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
