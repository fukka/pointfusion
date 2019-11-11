import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt

def show_pointcloud(filename):
	f = open(filename)
	l = f.readlines()
	print(len(l), len(l[0].split(',')))
	pc = np.ones((len(l[0].split(',')), 3))
	for i in range(3):
		lst = l[i].split(',')
		for j in range(len(lst)):
			pc[j, i] = float(lst[j])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
	f = open('pointcloud_temp1.out')
	l = f.readlines()
	print(len(l), len(l[0].split(',')))
	pc = np.ones((len(l[0].split(',')), 3))
	for i in range(3):
		lst = l[i].split(',')
		for j in range(len(lst)):
			pc[j, i] = float(lst[j])
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	o3d.visualization.draw_geometries([pcd])
	print(pc)

	# # xy = pc[:, 2]
	# plt.plot(pc[:, 0], pc[:, 1], 'r.')
	# plt.show()
