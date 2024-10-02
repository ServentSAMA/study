import numpy as np
import talib


def average_absolute_deviation(data):
    """
    计算数据序列的平均绝对偏差。

    参数:
    data (array-like): 一个数值列表或数组。

    返回:
    float: 平均绝对偏差值。
    """
    # 计算数据的平均值
    mean_value = np.mean(data)
    # 计算每个数据点与平均值的绝对偏差
    absolute_deviations = np.abs(data - mean_value)
    # 计算这些绝对偏差的平均值
    aad = np.mean(absolute_deviations)
    return aad


# 2024-01-16
# close = np.random.random(100)
# print(close)
# close = np.array([73.82, 74.89, 74.46, 74.07, 74, 73.29, 65.96])
close = np.array([72.85, 74.40, 74.35, 74.08, 73.63, 73.15, 68.87])
b = round((65.96 + 74.69 + 65.96) / 3, 2)
# b = 65.96
print(b)
# 计算收盘价的简单移动平均数 
# 默认是计算了 30 天的移动平均数 
output = talib.MA(close, 7)

print(close)
a = (b - output) / (0.015 * average_absolute_deviation(close))
# TYP := (IF(ISNULL(HIGH),CLOSE,HIGH) + IF(ISNULL(LOW),CLOSE,LOW) + CLOSE)/3;
# CCI:(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N)),RGB(0,240,0);

print(a)
