import argparse

import numpy as np

import matplotlib as mpl
from matplotlib import animation
from matplotlib import pyplot as plt

from scipy import spatial

def generate_circle(n_points):
	points = np.zeros((n_points,2))
	radius = np.sqrt( np.random.uniform(0, 1, n_points) )
	theta = np.random.uniform(0, 2*np.pi, n_points)
	points[:,0] = (radius * np.cos(theta))
	points[:,1] = (radius * np.sin(theta))
	return points

def generate_polygon(n_points):
	theta = np.linspace(0, 2*np.pi, n_points)
	xx = 2 * np.cos(theta)
	yy = 2 * np.sin(theta)
	points = np.zeros((n_points,2))
	points[:,0] = xx
	points[:,1] = yy
	return points



def main(n_points, n_steps, interval, generator, random_state):
	np.random.seed(random_state)
	if generator == "normal":
		points = np.random.normal(size=(n_points,2))
	elif generator == "uniform":
		points = np.random.uniform(size=(n_points,2))
	elif generator == "circle":
		points = generate_circle(n_points)
	elif generator == "polygon":
		points = generate_polygon(n_points)
	print "Points:", points.shape

	xmin, xmax = np.min(points[:,0]), np.max(points[:,0])
	ymin, ymax = np.min(points[:,1]), np.max(points[:,1])
	vmin, vmax = np.min([xmin,ymin]), np.max([xmax,ymax])
	vmin -= (vmax-vmin) * 0.25
	vmax += (vmax-vmin) * 0.25

	try:
		hull = spatial.ConvexHull(points)
		outlying_points = points[hull.vertices]
	except:
		outlying_points = points
	print "Outlying Points:", outlying_points.shape

	dmatrix = spatial.distance.pdist(outlying_points)
	dmax = np.nanmax(dmatrix)

	fig = plt.figure(figsize=(10,5))
	gs = mpl.gridspec.GridSpec(2, 4)

	circle_ax = fig.add_subplot(gs[0,:2], projection="polar")
	circle_ax.set_xticklabels(["0",
								"$\\frac{\\pi}{4}$",
								"$\\frac{\\pi}{2}$",
								"$\\frac{3\\pi}{4}$",
								"$\\pi$",
								"$\\frac{5\\pi}{4}$",
								"$\\frac{3\\pi}{2}$",
								"$\\frac{7\\pi}{4}$",])
	circle_ax.set_yticklabels([])
	circle_ax.set_ylim(0,1)
	polar_angle, = circle_ax.plot([0,0],[0,1], color="r", linewidth = 1)

	points_ax = fig.add_subplot(gs[0,2:],)
	points_ax.set_xlim(vmin, vmax)
	points_ax.set_xticks([])
	points_ax.set_ylim(vmin, vmax)
	points_ax.set_yticks([])
	points_ax.set_aspect("equal")
	points_ax.scatter(points[:,0], points[:,1], c="k", s=5)
	line1, = points_ax.plot([], [], color="b")
	line2, = points_ax.plot([], [], color="b")

	fun_ax = fig.add_subplot(gs[1, 1:3],)
	fun_ax.set_xlim(0,2*np.pi)
	fun_ax.set_xticks([0,
		np.pi/4,
		np.pi/2,
		3*np.pi/4,
		np.pi,
		5*np.pi/4,
		3*np.pi/2,
		7*np.pi/4,
		2*np.pi])
	fun_ax.set_xticklabels(["0",
		"$\\frac{\\pi}{4}$",
		"$\\frac{\\pi}{2}$",
		"$\\frac{3\\pi}{4}$",
		"$\\pi$",
		"$\\frac{5\\pi}{4}$",
		"$\\frac{3\\pi}{2}$",
		"$\\frac{7\\pi}{4}$",
		"$2\\pi$",])
	fun_ax.set_ylim(0,1.25*dmax)
	frames = np.linspace(0, 2*np.pi, n_steps)
	function, = fun_ax.plot(frames, np.zeros(n_steps), markersize=5,
			color="r")
	point, = fun_ax.plot(0, 0, "o", markersize=4, color="b")

	def FF(theta, points):
		v1 = np.zeros(points.shape)
		v2 = np.tile(np.asarray([np.cos(theta), np.sin(theta)]),
				(points.shape[0],1))
		dists = np.cross(v2-v1, points-v1) / np.linalg.norm(v2-v1)
		
		l1 = np.zeros((2,2))
		max_index = np.argmax(dists)
		max_point = points[max_index,:]
		l1[0,:] = np.asarray( [max_point[0]-np.cos(theta)*2*vmax,
			max_point[1]-np.sin(theta)*2*vmax] )
		l1[1,:] = np.asarray( [max_point[0]+np.cos(theta)*2*vmax,
			max_point[1]+np.sin(theta)*2*vmax] )

		l2 = np.zeros((2,2))
		min_index = np.argmin(dists)
		min_point = points[min_index,:]
		l2[0,:] = np.asarray( [min_point[0]-np.cos(theta)*2*vmax,
			min_point[1]-np.sin(theta)*2*vmax] )
		l2[1,:] = np.asarray( [min_point[0]+np.cos(theta)*2*vmax,
			min_point[1]+np.sin(theta)*2*vmax] )

		dist = np.cross(np.diff(l2, axis=0),
				max_point-l2[0,:])/np.linalg.norm(np.diff(l2, axis=0))
		return l1, l2, dist[0]

	def update(theta):
		polar_angle.set_data([theta, theta],[0,1])
		
		l1, l2, dist = FF(theta, outlying_points)
		line1.set_data(l1[:,0], l1[:,1])
		line2.set_data(l2[:,0], l2[:,1])
		
		index = np.where(frames==theta)
		yy = function.get_ydata()
		yy[index] = dist
		function.set_ydata(yy)
		
		point.set_ydata(dist)
		point.set_xdata(theta)
		return polar_angle, line1, line2, point

	def init():
		for theta in frames:
			res = update(theta)
		return res

	anim = animation.FuncAnimation(fig, update, init_func=init,
			frames=frames, interval=interval, blit=True)

	with open("video.html", "w") as fp:
		print >>fp, anim.to_html5_video()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parallel Lines",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("-p", "--n-points",
						type=int,
						default=20,
						metavar="int",
						help="number of points")
	parser.add_argument("-s", "--n-steps",
						type=float, default=36,
						metavar="float",
						help="angle step")
	parser.add_argument("-i", "--interval",
						type=int,
						default=50,
						metavar="int",
						help="update interval")
	parser.add_argument("-g", "--generator",
						default="normal",
						choices=["normal", "uniform", "circle", "polygon"])
	parser.add_argument("--random-state",
						type=int,
						default=None,
						metavar="int",
						help="seed used by the random number generator")

	arguments = vars(parser.parse_args())
	main(**arguments)
