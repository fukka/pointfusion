import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
# import matplotlib.pyplot as plt



def draw_box_matlab(ax, corners, color):
	for i in range(4):
		ax.plot([corners.T[i][0], corners.T[i + 4][0]],
		        [corners.T[i][1], corners.T[i + 4][1]],
		        color=color[2], linewidth=2)

	def draw_rect(selected_corners, color):
		prev = selected_corners[-1]
		for corner in selected_corners:
			ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2)
			prev = corner

	draw_rect(corners.T[:4], color[0])
	draw_rect(corners.T[4:], color[1])


def draw_box_cv2(image, corners, color):
	for i in range(4):
		start_point = (int(corners.T[i][0]), int(corners.T[i][1]))
		end_point = (int(corners.T[i + 4][0]), int(corners.T[i + 4][1]))
		cv2.line(image, start_point, end_point, color[2][::-1], 2)

	def draw_rect(selected_corners, color):
		prev = selected_corners[-1]
		for corner in selected_corners:
			start_point = (int(prev[0]), int(prev[1]))
			end_point = (int(corner[0]), int(corner[1]))
			cv2.line(image, start_point, end_point, color, 2)
			prev = corner

	draw_rect(corners.T[:4], color[0][::-1])
	draw_rect(corners.T[4:], color[1][::-1])
	return image



def show_annotation(nusc, sample_data_token):
	data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ANY)
	data = Image.open(data_path)
	_, ax = plt.subplots(1, 1, figsize=(9, 16))
	ax.imshow(data)
	for box in boxes:
		corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
		c = np.array(nusc.explorer.get_color(box.name)) / 255.0
		color = (c, c, c)
		draw_box_matlab(ax, corners, color)
		#
		# for i in range(4):
		# 	ax.plot([corners.T[i][0], corners.T[i + 4][0]],
		# 			[corners.T[i][1], corners.T[i + 4][1]],
		# 			color=color[2], linewidth=2)
		#
		# def draw_rect(selected_corners, color):
		# 	prev = selected_corners[-1]
		# 	for corner in selected_corners:
		# 		ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2)
		# 		prev = corner
		#
		# draw_rect(corners.T[:4], color[0])
		# draw_rect(corners.T[4:], color[1])

def show_annotation_cv2(nusc, sample_data_token):
	data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ANY)
	image = cv2.imread(data_path)
	u, v = image.shape[:2]

	for box in boxes:
		corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
		if (corners[0, :] > v).any() or (corners[1, :] > u).any()\
				or (corners[0, :] < 0).any() or (corners[1, :] < 0).any():
			print(corners[0, :], corners[1, :])
			continue
		c = nusc.explorer.get_color(box.name)
		color = (c, c, c)
		image = draw_box_cv2(image, corners, color)
		# for i in range(4):
		# 	start_point = (int(corners.T[i][0]), int(corners.T[i][1]))
		# 	end_point = (int(corners.T[i + 4][0]), int(corners.T[i + 4][1]))
		# 	cv2.line(image, start_point, end_point, color[2][::-1], 2)
		#
		# def draw_rect(selected_corners, color):
		# 	prev = selected_corners[-1]
		# 	for corner in selected_corners:
		# 		start_point = (int(prev[0]), int(prev[1]))
		# 		end_point = (int(corner[0]), int(corner[1]))
		# 		cv2.line(image, start_point, end_point, color, 2)
		# 		prev = corner
		#
		# draw_rect(corners.T[:4], color[0][::-1])
		# draw_rect(corners.T[4:], color[1][::-1])
	cv2.imwrite('img.jpg', image)
	return image


def warpPointcloud(pointcloud):
	new_shape = np.zeros((4, pointcloud.shape[0]))
	for i in range(pointcloud.shape[0]):
		new_shape[:3, i] = pointcloud[i, :]
	return LidarPointCloud(new_shape)


def lidarCoord2rgbCoord(pc, nusc, camera_token, pointsensor_token):
	cam = nusc.get('sample_data', camera_token)
	pointsensor = nusc.get('sample_data', pointsensor_token)

	# Points live in the point sensor frame. So they need to be transformed via global to the image plane.
	# First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
	cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
	pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
	pc.translate(np.array(cs_record['translation']))

	# Second step: transform to the global frame.
	poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
	pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
	pc.translate(np.array(poserecord['translation']))

	# Third step: transform into the ego vehicle frame for the timestamp of the image.
	poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
	pc.translate(-np.array(poserecord['translation']))
	pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

	# Fourth step: transform into the camera.
	cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
	pc.translate(-np.array(cs_record['translation']))
	pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

	# Fifth step: actually take a "picture" of the point cloud.
	# Grab the depths (camera frame z axis points away from the camera).
	depths = pc.points[2, :]

	coloring = depths

	# Take the actual picture (matrix multiplication with camera-matrix + renormalization).
	points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

	# # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
	# # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
	# # casing for non-keyframes which are slightly out of sync.
	# mask = np.ones(depths.shape[0], dtype=bool)
	# mask = np.logical_and(mask, depths > min_dist)
	# mask = np.logical_and(mask, points[0, :] > 1)
	# mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
	# mask = np.logical_and(mask, points[1, :] > 1)
	# mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
	# points = points[:, mask]
	# coloring = coloring[mask]

	return points, coloring


def resizeRGB(im, index):
	u, v = im.shape[2:]
	new_img = np.zeros((u, v, 3))
	for u_i in range(u):
		for v_i in range(v):
			new_img[u_i, v_i, :] = im[index, :,u_i, v_i]
	return new_img

def show_pointcloud(filename):
	import open3d as o3d
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
	show_pointcloud('pointcloud_temp3.out')
	print("finished")

	# # xy = pc[:, 2]
	# plt.plot(pc[:, 0], pc[:, 1], 'r.')
	# plt.show()
