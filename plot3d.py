from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

x_line = np.linspace(-10, 10, 70)
y_line = np.linspace(-10, 10, 70)

X, Y = np.meshgrid(x_line, y_line)


# Z = function(X,Y)

Z = np.square(X-Y) + 100*np.sin(X+Y) + 10*np.cos(X-Y)
# Z = X + Y

# ax.plot_wireframe(X, Y, Z, color='green')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')

plt.show()