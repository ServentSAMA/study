import pandas as pd
import os
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,alley,price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


data = pd.read_csv(data_file)

print(data)
# iloc = index location
# 通过位置索引iloc函数，将data数据分成前两列，和最后一列
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# print(inputs.iloc[:, 0])
print(inputs.shape)
# 处理缺失的值
# pandas包是python中常用的数据分析工具，pandas可以与张量兼容
inputs = inputs.fillna(inputs.mean())

print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)



X, y = torch.tensor(inputs.to_numpy(dtype=float)), torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(y)

# 借鉴评论的方法
def drop_col(data):

    data_sum = data.isna().sum() # 统计缺失值
    print(data_sum)
    data_dict = data_sum.to_dict() # 将统计结果转换成字典
    print(data_sum.to_dict())
    data_max = max(data_dict, key=data_dict.get) # 获取缺失值最大的列
    del data[data_max] # 删除缺失值最大的列

drop_col(data)
print(data)
