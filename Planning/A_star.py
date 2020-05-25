import sys
import math
import numpy as np
from heapq import heapify, heappush, heappop
# from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

plt.style.use('seaborn-pastel')

# Sent for figure
font = {'size'   : 8}
matplotlib.rc('font', **font)


x_lim = (0,120)
y_lim = (0,120)

n_iterations = 500
interval = 50

openSet = []
heapify(openSet)


goal = (100,108,6)
start = np.array([4,4])
current = np.array([start[0],start[1]])
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

for i in range(len(obstacles)):
	obstacles[i] = (obstacles[i][0]*2 + 60, obstacles[i][1]*2+60, obstacles[i][2]*2)

# obstacles = [(i,j,2) for i in range(-25,25,8) for j in range(-25,25,8)]

obs_np = list(map(np.array,obstacles))

goal_reached = False


a = []
b = []

visited = np.array([[0 for i in range(120)] for j in range(120)])

stack = []
stack.append(current)

ds = [np.array([1,1]),np.array([-1,-1]),np.array([-1,1]),np.array([1,-1])]

def h(pos):
	return np.linalg.norm(pos - np.array([goal[:2]]))


def goal_check(pos):
	distance = np.linalg.norm(pos - np.array([goal[:2]]))
	print(pos,goal[2],distance)
	if distance < goal[2]:
		return True
	else:
		return False

def free(pos):
	good = True
	for o in obstacles:
		distance = np.linalg.norm(pos - np.array([o[:2]]))
		if distance < o[2]:
			good = False
			break
	return good

def update(i):
	global current, openSet, fScore, gScore
	print("Update:",i)

	if len(openSet) > 0:
		current = heappop(openSet)[1]
		a.append(current[0])
		b.append(current[1])
	if goal_check(current):
		print("Goal Reached!")
		sys.exit()
	else:

		for d in ds:
			nextNode = current+d
			if (x_lim[0] < nextNode[0] <  x_lim[1]) and (y_lim[0] < nextNode[1] <  y_lim[1]) and visited[nextNode[0],nextNode[1]] == 0 and free(nextNode):
				
				tentative_gScore = gScore[current[0],current[1]] + np.sqrt(1)
				
				if tentative_gScore < gScore[nextNode[0],nextNode[1]]:
					gScore[nextNode[0],nextNode[1]] = tentative_gScore
					fScore[nextNode[0],nextNode[1]] = gScore[nextNode[0],nextNode[1]] + h(nextNode)

					if nextNode not in openSet:
						heappush(openSet,(fScore[nextNode[0],nextNode[1]], nextNode))


		graph.set_data(a, b)
	
	return graph,



if __name__ == "__main__":
	global fig, fScore, gScore

	fScore = np.ones(shape=(120,120))*math.inf
	gScore = np.ones(shape=(120,120))*math.inf

	gScore[start[0],start[1]] = 0
	fScore[start[0],start[1]] = h(start)

	fig = plt.figure(figsize=(8,8))
	ax1 = subplot2grid((1,1),(0,0))

	graph, = ax1.plot([],[],'o',markersize='2.0',color='blue')

	margin = 5.0
	ax1.set_xlim(0,120)
	ax1.set_ylim(0,120)
	ax1.set_xlabel("M-axis (Line Slope)")
	ax1.set_ylabel("C-axis (Line Y-Intercept)")

	ax1.grid(False)


	for o in obstacles:
		circle = plt.Circle((o[0], o[1]), o[2], color='r')
		ax1.add_artist(circle)

	goalCircle = plt.Circle((goal[0],goal[1]),goal[2],color='black')
	ax1.add_artist(goalCircle)

	startCircle = plt.Circle((start[0],start[1]),4,color='green')
	ax1.add_artist(startCircle)

	x = np.arange(0,120,1)
	y = np.arange(0,120,1)

	X,Y = np.meshgrid(x,y)

	scat = plt.scatter(X,Y,s=0.3,color='black')

	simulation = FuncAnimation(fig, update, blit=True, frames=n_iterations, interval=interval, repeat=False)
	plt.show()
