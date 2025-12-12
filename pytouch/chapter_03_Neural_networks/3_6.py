"""
softmax 从零开始实现
"""
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 图像输入的大小
num_inputs = 784
# 输出大小
num_outputs = 10
# 权重
W = torch.normal(0, 0.01, size=(num_inputs,num_outputs), requires_grad=True)
# 偏移量
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作
X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(X.sum(0, keepdim=True),X.sum(1,keepdim=True))

# 定义softmax函数 
def softmax(X):
    # exp对每个元素做指数运算
    # e（自然对数）做底数，矩阵值作为指数计算
    X_exp = torch.exp(X)
    # axis为1，是以每一行进行求和，keepdim表示不对形状进行处理
    partition = X_exp.sum(1, keepdim=True)
    # 使用广播机制对矩阵中的值进行相除
    return X_exp / partition 

# 使用正态分布生成0-1的2*5的tensor矩阵
X = torch.normal(0, 1, (2, 5))
print(X)
X_prob = softmax(X)

print(X_prob,X_prob.sum(1))

# 实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0, 1], y]中
# [0,1]表示取第0,1行，y=[0,2]表示取第一行第一个，第二行第3个（从零开始）
print(y_hat[[0, 1], y])

# 定义交叉熵损失函数
def cross_entropy(y_hat, y):

    return -torch.log(y_hat[range(len(y_hat)),y])

print(cross_entropy(y_hat, y))

# 计算分类精度
def accuracy(y_hat, y):
    '''计算预测正确的数量'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat,y) / len(y))



def evaluate_accuracy(net, data_iter):
    '''计算在指定数据集上模型的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式，不计算梯度
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y), y.numel())
    return metric[0] / metric[1]

# print(ty)

def train_epoch_ch3(net, train_iter, loss, updater):
    # 如果是torch模块的，则使用torch的训练
    if isinstance(net, torch.nn.Module):
        net.train()
    # 对数据进行叠加
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    # 扫描数据集
    for X, y in train_iter:
        # X是训练集
        # y是正确值
        # 定义模型
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        # 如果是torch的优化器（Optimizer）
        # Optimizer有很多实现：SGD、Adam
        if isinstance(updater, torch.optim.Optimizer):
            # zero_grad是Optimizer模块中的一个函数，用于将优化器中所有参数的梯度置为零
            # 以避免上一次的梯度信息对当前的参数更新产生影响
            updater.zero_grad()
            # 计算梯度信息
            l.backward()
            # 根据计算出的梯度信息更新模型参数
            updater.step()

            metric.add(float(1) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # 使用自定义的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 训练损失 / 总数，训练精度 / 总数
    return metric[0] / metric[2],metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3,0.9],
                            legend=['train loss','train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics

    # assert断言关键字
    # 判断某个条件是否满足，如果不满足，就会抛出一个 AssertionError 异常并终止程序的执行
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 使用3.2节中定义的小批量随机梯度下降来优化模型的随时函数，设置学习率为0.1
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10

# 在交互式窗口中运行
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 对图像进行分类预测
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

# 直接运行是没有训练过的，先运行train_ch3函数，再运行下方代码才会准确预测
# predict_ch3(net, test_iter)


