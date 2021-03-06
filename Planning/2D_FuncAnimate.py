import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
points, = ax.plot(np.random.rand(10), 'o')
ax.set_ylim(0, 1)

def update(data):
    points.set_data(data)
    return points,

def generate_points():
    while True:
        yield np.random.rand(10)  # change this



ani = animation.FuncAnimation(fig, update, generate_points, interval=30)
plt.show()