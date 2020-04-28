import sys
import math
import numpy as np
# from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

plt.style.use('seaborn-pastel')

# Sent for figure
font = {'size'   : 8}
matplotlib.rc('font', **font)


x_lim = (-30,30)
y_lim = (-30,30)

low = -100
high = 100

init_low_x = -20
init_high_x = -19
init_low_y = -20
init_high_y = -19


# print(rand1)
n_particles = 5

n_iterations = 300
interval = 30

goal = (20,24,3)
obstacles = [	(-15,10,5),
				(-3,16,5),
				(17,5,2),
				(1,-3,4),
				(0,-20,6),
				(20,12,4),
				(-14,-18,4),
				(13,-12,4),
				(-10,-8,4),
				(14,-20,2),
				(8,5,4),
				(-22,-3,4),
				(10,20,3),
				(23,-2,4)
			]
obs_np = list(map(np.array,obstacles))
# print(obs_np)

pdots_x = [0 for i in range(n_particles)]
pdots_y = [0 for i in range(n_particles)]
lines = []
guides = []
particles = []


g_value = 1000000
g_position = np.array([low + 2*high*np.random.random(), low + 2*high*np.random.random()])
error = []



def computeField(particle):
	p_pos = particle.position[-1]
	print(particle.position[-1])
	x = p_pos[0]
	y = p_pos[1]
	energy = ((x-goal[0])**2 + (y-goal[1])**2)*200
	for o in obstacles:
		energy -= ((x-o[0])**2 + (y-o[1])**2)*2
	# rastrigin -= ((x-x_lim[0])**2 + (y-y_lim[0])**2 + (x-x_lim[1])**2 + (y-y_lim[1]))/200
	return energy

def rastrigin(pos):
	# paraboloid = pos[0]**2 + pos[1]**2
	x = pos[0]+10
	y = pos[1]+20
	rastrigin = 20 + (x)**2 + (y)**2 - 10*np.cos(2*np.pi*(x)) - 10*np.cos(2*np.pi*y)
	return rastrigin

def update_velocity(particle, w0, w1, w2):
	global g_value, g_position
	for o in obs_np:
		distance = np.linalg.norm(particle.position[-1] - np.array([o[:2]]))
		r = o[2] + 1.5
		if distance < r:
			new_velocity = (particle.position[-1] - np.array([o[:2]]) + np.random.normal(size=(1,2)) )/3
			return new_velocity

		new_velocity =	w0*particle.velocity + \
						w1*(particle.b_position - particle.position[-1]) + \
						w2*(g_position - particle.position[-1])

	return new_velocity

def update_position(particle):
	p_vel = particle.velocity
	new_position = particle.position[-1] + p_vel
	if new_position.shape[0]==1:
		new_position = new_position[0]
	if p_vel.shape[0]==1:
		p_vel = p_vel[0]
	if new_position[0] < x_lim[0]:
		new_position[0] = x_lim[0]
		p_vel[0] *= -1
	if new_position[0] > x_lim[1]:
		new_position[0] = x_lim[1]
		p_vel[0] *= -1
	if new_position[1] < y_lim[0]:
		new_position[1] = y_lim[0]
		p_vel[1] *= -1
	if new_position[1] > y_lim[1]:
		new_position[1] = y_lim[1]
		p_vel[1] *= -1
	return new_position


class Particle:
	def __init__(self, id):
		self.id = id
		self.position = np.array([[init_low_x+ 2*init_high_x*np.random.random(), init_low_y + 2*init_high_y*np.random.random()]])
		self.velocity = np.array([0.0, 0.0])
		self.b_value = math.inf
		self.b_position = self.position[-1]

	def __str__(self):
		return "Particle {}: Position:{} Velocity:{} bVal:{} bPos{}".format(self.id,self.position,self.velocity,self.b_value,self.b_position)

def update(i):
	global data, g_value, g_position, error, line2, lines, guides
	for k in range(n_particles):	

		print(particles[k].position.shape)

		f = computeField(particles[k])
		if f < particles[k].b_value:
			particles[k].b_value = f
			particles[k].b_position = particles[k].position[-1]

		if f < g_value:
			g_value = f
			g_position = particles[k].position[-1]

		particles[k].velocity = update_velocity(particles[k], w0, w1, w2)	
		new_particle_position = update_position(particles[k])
		particles[k].position = np.vstack([particles[k].position, new_particle_position])

		a = particles[k].position[:,0]
		b = particles[k].position[:,1]

		lines[k].set_data(a,b)

	time = np.linspace(1,i+2,i+2)
	error.append(g_value)

	line2.set_data(time, np.asarray(error)/1000)
	return lines[0], line2, tuple(guides)



if __name__ == "__main__":
	
	best = (0.9, 0.11, 0.8)
	demo = (0.95, 0.001, 0.05)
	w0, w1, w2 = demo
	m = 9

	fig = plt.figure(figsize=(16,8))
	ax1 = subplot2grid((1,2),(0,0))

	for i in range(n_particles):
		line_i, = ax1.plot([], [], lw=1)
		lines.append(line_i)

	ax2 = subplot2grid((1,2),(0,1))
	line2, = ax2.plot([],[],lw=1)

	margin = 5.0
	ax1.set_xlim(-30,30)
	ax1.set_ylim(-30,30)
	ax1.set_xlabel("M-axis (Line Slope)")
	ax1.set_ylabel("C-axis (Line Y-Intercept)")
	
	ax2.set_xlim(0,200)
	ax2.set_ylim(-100,200)
	ax2.set_xlabel("Iterations Elapsed (PSO)")
	ax2.set_ylabel("Total Residual (PSO)")

	ax1.grid(False)
	ax2.grid(True)

	x = np.arange(-30,30,2)
	y = np.arange(-30,30,2)
	X,Y = np.meshgrid(x,y)
	u = ((X-goal[0])**2)*10 + 1
	v = ((Y-goal[1])**2)*10 + 1

	ax1.quiver(X,Y,u,v)

	for o in obstacles:
		circle = plt.Circle((o[0], o[1]), o[2], color='r')
		ax1.add_artist(circle)

	goalCircle = plt.Circle((goal[0],goal[1]),goal[2],color='g')
	ax1.add_artist(goalCircle)

	for k in range(n_particles):
		particles.append(Particle(k))

	simulation = FuncAnimation(fig, update, blit=False, frames=n_iterations, interval=interval, repeat=False)
	plt.show()
