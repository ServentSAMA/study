# 读取文本文件
import os

mysqld = os.open('mysqld.log', flags=os.O_RDONLY)

# 读取一行
line = os.read(mysqld, 1024)
print(line.decode('utf-8'))
# 读取每一行
# while True:
#     line = os.read(mysqld, 1024)
#     if not line:
#         break
#     print(line.decode('utf-8'))



