usernames = ['zhangsan', 'lisi', 'wangwu']


def get_username(username):
    username.append('张三')


get_username(usernames)

print(usernames)

'''
使用任意数量的关键字实参
'''


def build_profile(first, last, **user_info):
    user_info['first'] = first
    user_info['last'] = last

    return user_info


user_info = build_profile('shen', 'wenjie', location='a', field='b')

print(user_info)
