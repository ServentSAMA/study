"""
7.7. 稠密连接网络（DenseNet）
"""
import torch
import torch.nn as nn
import d2l.torch as d2l


def conv_block(input_channels, num_channels):
    """
    7.7.2. 稠密块体
    input_channels: 当前 block 的输入通道数（即前一层输出的通道数）。
    num_channels: 当前 block 中每个残差单元的输出通道数（也是主路径中卷积层的输出通道数）。
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)


# Y.shape

def transition_block(input_channels, num_channels):
    """
        7.7.3 过渡层
        由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。 而过渡层可以用来控制模型复杂度。
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


'''
    7.7.4 7.7.4. DenseNet模型
    DenseNet首先使用同ResNet一样的单卷积层和最大汇聚层。
'''
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

'''
类似于ResNet使用的4个残差块，DenseNet使用的是4个稠密块。 与ResNet类似，我们可以设置每个稠密块使用多少个卷积层。 
这里我们设成4，从而与 7.6节的ResNet-18保持一致。 
稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。
'''
# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))

X = torch.rand(size=(1, 1, 96, 96))
for blk in net:
    if isinstance(blk, nn.Sequential):
        # print(blk.__class__.__name__)
        for model in blk:
            X = model(X)
            if model.__class__.__name__ in ["Conv2d", "Linear"]:
                print(model.__class__.__name__, 'output shape:\t', X.shape)
                pass
    else:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
