"""
5.5. 读写文件
"""
import torch
from torch import nn
from torch.nn import functional as F
'''
5.5.1. 加载和保存张量
'''
# 保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

# 保存张量列表
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

# 保存字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
'''
5.5.2. 加载和保存模型参数
'''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()


X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

Y_clone = clone(X)
print(Y_clone == Y)

# 1、即使不需要将经过训练的模型部署到不同的设备上，存储模型参数还有什么实际的好处？
#    可以生成模型参数的

# 官方
