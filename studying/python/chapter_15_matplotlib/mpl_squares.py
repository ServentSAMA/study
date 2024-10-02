import matplotlib.pyplot as plt

squares = [1,4,9,16,25]
fig,ax = plt.subplots()
ax.plot(squares)

ax.set_title("squares",fontsize=24)

plt.show()