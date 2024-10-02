import datetime
import time
import keyword

"""
"""

if __name__ == '__main__':
    print(int(time.time()))
    print(keyword.kwlist)

    print(bool(False))
    # 0数字
    print(bool(0))
    print(bool(0.0))
    print(bool(None))
    # 空字符串
    print(bool(""))
    print(bool(''))
    # 空列表
    print("空列表")
    print(bool([]))
    print(bool(list()))
    # 空元组
    print("空元组")
    print(bool(()))
    print(bool(tuple()))
    # 空字典
    print("空字典")
    print({})
    print(dict())

    print(set())
    lst = map(str, [i for i in range(10)])
    print(list(lst))

    print("type类型检查函数")
    print(type("字符串"))
