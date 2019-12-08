import os
import numpy as np
from nuscenes import NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def get_pointcloud(nusc, bottom_left, top_right, box, pointsensor_token: str, camera_token: str,
                   min_dist: float = 1.0) -> Tuple:
	"""
	Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
	plane.
	:param pointsensor_token: Lidar/radar sample_data token.
	:param camera_token: Camera sample_data token.
	:param min_dist: Distance from the camera below which points are discarded.
	:return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
	"""
	sample_data = nusc.get("sample_data", camera_token)
	explorer = NuScenesExplorer(nusc)

	cam = nusc.get('sample_data', camera_token)
	pointsensor = nusc.get('sample_data', pointsensor_token)

	im = Image.open(osp.join(nusc.dataroot, cam['filename']))

	sample_rec = explorer.nusc.get('sample', pointsensor['sample_token'])
	chan = pointsensor['channel']
	ref_chan = 'LIDAR_TOP'
	# pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps = 10)
	file_name = os.path.join(nusc.dataroot, nusc.get('sample_data', sample_rec['data'][chan])['filename'])
	pc = LidarPointCloud.from_file(file_name)
	data_path, boxes, camera_intrinsic = nusc.get_sample_data(pointsensor_token, selected_anntokens=[box.token])
	pcl_box = boxes[0]

	original_points = pc.points.copy()

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
	depths = pc.points[2, :]

	# Take the actual picture (matrix multiplication with camera-matrix + renormalization).
	points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
	center = np.array([[box.center[0]], [box.center[1]], [box.center[2]]])
	box_center = view_points(center, np.array(cs_record['camera_intrinsic']), normalize=True)
	box_corners = box.corners()



	image_center = np.asarray([((bottom_left[0] + top_right[0]) / 2), ((bottom_left[1] + top_right[1]) / 2), 1])

	# rotate to make the ray passing through the image_center into the z axis
	z_axis = np.linalg.lstsq(np.array(cs_record['camera_intrinsic']), image_center, rcond=None)[0]
	v = np.asarray([z_axis[0], z_axis[1], z_axis[2]])
	z = np.asarray([0., 0., 1.])
	normal = np.cross(v, z)
	theta = np.arccos(np.dot(v, z) / np.sqrt(v.dot(v)))
	new_pts = []
	new_corner = []
	old_pts = []
	points_3 = points_3[:3, :]
	translate = np.dot(rotation_matrix(normal, theta), image_center)
	for point in points_3.T:
		new_pts = new_pts + [np.dot(rotation_matrix(normal, theta), point)]
	for corner in box_corners.T:
		new_corner = new_corner + [np.dot(rotation_matrix(normal, theta), corner)]

	points = np.asarray(new_pts)
	original_points = original_points[:3, :].T
	new_corners = np.asarray(new_corner)

	reverse_matrix = rotation_matrix(normal, -theta)

	# Sample 400 points
	if np.shape(new_pts)[0] > 400:
		mask = np.random.choice(points.shape[0], 400, replace=False)
		points = points[mask, :]
		original_points = original_points[mask, :]

	shift = np.expand_dims(np.mean(points, axis=0), 0)
	points = points - shift  # center
	new_corners = new_corners - shift  # center
	dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
	points = points / dist  # scale
	new_corners = new_corners / dist  # scale

	# Compute offset from point to corner
	n = np.shape(points)[0]
	offset = np.zeros((n, 8, 3))
	for i in range(0, n):
		for j in range(0, 8):
			offset[i][j] = new_corners[j] - points[i]

	# Compute mask on which points lie inside of the box
	m = []
	for point in original_points:
		if in_box(point, pcl_box.corners()) == True:
			m = m + [1]
		else:
			m = m + [0]
	m = np.asarray(m)

	return points.T, m, offset, reverse_matrix, new_corners