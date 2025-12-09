import numpy as np
# 一维数组
a = np.array([1, 2, 3])
print(a)  # 输出: [1 2 3]

# 二维数组
b = np.array([[1, 2], [3, 4]])
print(b)  # 输出: [[1 2]
          #        [3 4]]
# 全零数组
zeros = np.zeros((2, 3))  # 2行3列的全零数组
print(zeros)

# 全一数组
ones = np.ones((3, 2))  # 3行2列的全一数组
print(ones)

# 空数组（未初始化）
empty = np.empty((2, 2))
print(empty)

# 范围数组
range_array = np.arange(0, 10, 2)  # 从0到10（不包括10），步长为2
print(range_array)  # 输出: [0 2 4 6 8]

# 等间隔数组
linspace_array = np.linspace(0, 1, 5)  # 在0到1之间生成5个等间距的数
print(linspace_array)  # 输出: [0.   0.25 0.5  0.75 1.  ]
# 改变数组形状
c = np.arange(6)
d = c.reshape(2, 3)  # 将一维数组转换为2行3列的二维数组
print(d)
# 水平拼接
e = np.array([[1, 2], [3, 4]])
f = np.array([[5, 6], [7, 8]])
hstacked = np.hstack((e, f))
print(hstacked)

# 垂直拼接
vstacked = np.vstack((e, f))
print(vstacked)
# 水平分割
g = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
h_split = np.hsplit(g, 2)  # 分成两部分
print(h_split)

# 垂直分割
v_split = np.vsplit(g, 2)
print(v_split)
# 获取单个元素
arr = np.array([1, 2, 3, 4])
print(arr[0])  # 输出: 1

# 多维数组索引
multi_arr = np.array([[1, 2], [3, 4]])
print(multi_arr[0, 1])  # 输出: 2
# 一维数组切片
sliced = arr[1:3]
print(sliced)  # 输出: [2 3]

# 多维数组切片
sliced_multi = multi_arr[:, 1]  # 取所有行的第2列
print(sliced_multi)  # 输出: [2 4]
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 加法
add_result = x + y
print(add_result)  # 输出: [5 7 9]

# 减法
sub_result = x - y
print(sub_result)  # 输出: [-3 -3 -3]

# 乘法
mul_result = x * y
print(mul_result)  # 输出: [4 10 18]

# 除法
div_result = x / y
print(div_result)  # 输出: [0.25 0.4  0.5 ]
# 标量广播
broadcast = x + 10
print(broadcast)  # 输出: [11 12 13]

stats_arr = np.array([[1, 2], [3, 4]])

# 最大值
max_val = stats_arr.max()
print(max_val)  # 输出: 4

# 最小值
min_val = stats_arr.min()
print(min_val)  # 输出: 1

# 求和
sum_val = stats_arr.sum()
print(sum_val)  # 输出: 10

# 平均值
mean_val = stats_arr.mean()
print(mean_val)  # 输出: 2.5

# 标准差
std_dev = stats_arr.std()
print(std_dev)  # 输出: 1.118033988749895

# 生成随机浮点数 (0到1之间)
random_float = np.random.rand(2, 3)
print(random_float)

# 生成随机整数 (指定范围)
random_int = np.random.randint(1, 10, size=(2, 2))
print(random_int)
