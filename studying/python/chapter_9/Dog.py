class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sit(self):
        """模拟小狗收到命令蹲下"""
        print(f'{self.name}，现在蹲下了')

    def roll_over(self):
        """模拟打滚"""
        print(f'{self.name}，现在在打滚')


jack = Dog("jack", 6)

print(jack.name)
print(jack.age)

jack.sit()
jack.roll_over()
