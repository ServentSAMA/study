"""
8.5. 循环神经网络的从零开始实现
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


'''
8.5.1. 独热编码
'''
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape)

'''
8.5.2 初始化模型参数
'''
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

'''
8.5.3 循环神经网络模型
'''
#
def init_run_state(batch_size, num_hiddens, device):
    """
        为了定义循环神经网络模型， 我们首先需要一个init_rnn_state函数在初始化时返回隐状态。
        这个函数的返回是一个张量，张量全用0填充， 形状为（批量大小，隐藏单元数）。
    """
    # 形状为（批量大小，隐藏单元数）
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    output = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        output.append(Y)
    return torch.cat(output, dim=0), (H,)


class RNNModelScratch:
    """
       定义了所有需要的函数之后，接下来我们创建一个类来包装这些函数，
       并存储从零开始实现的循环神经网络模型的参数。
    """
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_run_state, rnn)

state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)

'''
8.5.4. 预测
    让我们首先定义预测函数来生成prefix之后的新字符， 其中的prefix是一个用户提供的包含多个字符的字符串。 
    在循环遍历prefix中的开始字符时， 我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。 
    这被称为预热（warm-up）期， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。 
    预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们。
'''

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

'''
8.5.5. 梯度裁剪
'''