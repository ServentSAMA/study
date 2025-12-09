import pandas as pd

dataFrame = pd.read_excel('C:\\Users\\Shen\\Desktop\\项目\\WMS\\浩大\\北洋佳美库存(1).xlsx')

dataFrame = dataFrame.dropna(axis=1)

print(dataFrame)

