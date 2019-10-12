import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
plt.style.use('seaborn-pastel')

low = -10
high = 10

init_low_x = -10
init_high_x = 10
init_low_y = -10
init_high_y = 10


f  = lambda x, y: 20 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
xmin, xmax, xstep = low, high, 0.01
ymin, ymax, ystep = low, high, 0.01
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f(x, y)



fig = plt.figure()
ax = plt.axes(xlim=(low, high), ylim=(low, high))
ax.contour(x, y, z, levels=np.linspace(0, 200, 30), norm=LogNorm(), cmap=plt.cm.jet)

lines = []
particles = []

n_particles = 10

n_iterations = 1000
interval = 50

g_value = 1000000
g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])
k = 0
w1, w2 = 0.5, 0.5

for i in range(n_particles):
	line_i, = ax.plot([], [], lw=4)
	lines.append(line_i)


def energy(pos):
	# paraboloid = pos[0]**2 + pos[1]**2
	rastrigin = 20 + pos[0]**2 + pos[1]**2 - 10*np.cos(2*np.pi*pos[0]) - 10*np.cos(2*np.pi*pos[1])
	return rastrigin

def init():
    return tuple(lines)

def animate(i):
	global g_value, g_position
	for k in range(n_particles):	

		# Calculate target value
		f = energy(particles[k].position[-1])

		# Compare against pv and gv and replace if needed
		if f < particles[k].b_value:
			particles[k].b_value = f
			particles[k].b_position = particles[k].position[-1]

		if f < g_value:
			g_value = f
			g_position = particles[k].position[-1]

		# calculate new velocity
		particles[k].velocity = 0.9*particles[k].velocity + \
								0.01*(particles[k].b_position - particles[k].position[-1]) + \
								0.09*(g_position - particles[k].position[-1])			

		# move
		new_particle_position = particles[k].position[-1] + particles[k].velocity

		# append new position
		particles[k].position = np.vstack([particles[k].position, new_particle_position])

		a = particles[k].position[:,0]
		b = particles[k].position[:,1]

		print("X:",a)
		print("Y:",b)

		lines[k].set_data(a,b)

	return tuple(lines)


class Particle:
	def __init__(self, id):
		self.id = id
		self.position = np.array([[init_low_x+ 2*init_high_x*np.random.random(), init_low_y + 2*init_high_y*np.random.random()]])
		self.velocity = np.array([2*np.random.random(), 2*np.random.random()])
		self.b_value = math.inf
		self.b_position = self.position[-1]

	def __str__(self):
		return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id,
																			self.position, 
																			self.velocity, 
																			self.b_value, 
																			self.b_position)




if __name__ == "__main__":

	for i in range(n_particles):
		particles.append(Particle(i))

	anim = FuncAnimation(fig, animate, init_func=init,
                               frames=n_iterations, interval=interval, blit=True)

	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.show()
