alien_0 = {'color': 'green', 'points': 5}

print(alien_0['color'])
print(alien_0['points'])
# 访问不存在的键时会报键错误
# print(alien_0['a'])

print(alien_0)

alien_0['x_position'] = 25
alien_0['y_position'] = 10

print(alien_0)


for k, v in alien_0.items():
    print(k)
    print(v)