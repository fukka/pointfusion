import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from nuscenes import NuScenes
from nuscenes import NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility

def get_pcl():
        # v1.0-trainval
        # nusc = Nuscenes(version='v1.0-trainval', dataroot='/home/fengjia/data/sets/nuscenes', verbose=True)
	nusc = NuScenes(version='v1.0-trainval', dataroot='/home/fengjia/data/sets/nuscenes', verbose=True)
	f = open(r'annotations_list.txt', 'w')

	count = 0
	for scene in nusc.scene:
		sample_token = scene['first_sample_token']
		my_sample = nusc.get('sample', sample_token)
		while sample_token != '':
			my_sample = nusc.get('sample', sample_token)
			for i in range(len(my_sample['anns'])):
				my_annotation_token = my_sample['anns'][i]
				my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
				my_sample_token = my_annotation_metadata['sample_token']
				my_sample_temp = nusc.get('sample', my_sample_token)
				sample_data_cam = nusc.get('sample_data', my_sample_temp['data']['CAM_FRONT'])
				s = sample_data_cam['token']
				s += '_'
				s += my_annotation_metadata['token']
				s += '\n'
				f.write(s)
				count += 1
			sample_token = my_sample['next']
	f.close()
	print(count)


def display(pc, i):
	# 3D plotting of point cloud
	fig=plt.figure(i)
	ax = fig.gca(projection='3d')

	#ax.set_aspect('equal')
	X = pc[0]
	Y = pc[1]
	Z = pc[2]
	c = pc[3]

	"""
	radius = np.sqrt(X**2 + Y**2 + Z**2)
	X = X[np.where(radius<20)]
	Y = Y[np.where(radius<20)]
	Z = Z[np.where(radius<20)]
	c = pc.points[3][np.where(radius<20)]
	print(radius)
	"""
	ax.scatter(X, Y, Z, s=1, c=cm.hot((c/100)))

	max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
	Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
	Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
	Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

	i=0
	for xb, yb, zb in zip(Xb, Yb, Zb):
		i = i+1
		ax.plot([xb], [yb], [zb], 'w')

get_pcl()
