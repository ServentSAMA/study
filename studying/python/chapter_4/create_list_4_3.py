for value in range(10):
    print(value)

list_1 = list(range(5))
print(list_1)
list_1 = list(range(1,10,2))
print(list_1)

# 练习4.3
for value in range(1,21):
    print(value)

list_2 = [value for value in range(1,1000001)]

print(sum(list_2))




