
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from imageio import imread 
import torch

import numpy as np
import numpy.random as npr
from PIL import Image
import random
import time
import pdb
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes import NuScenesExplorer 
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import os 

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from mpl_toolkits.mplot3d import Axes3D

from utils import get_pointcloud 

import cv2 


if __name__ == '__main__':
    data_path = "/home/fengjia/data/sets/nuscenes"
    save_path = "/home/fengjia/data/sets/nuscenes_local"
    if not os.path.isdir(save_path):
        print('save path not exist')
        exit(0)

    nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
    explorer = NuScenesExplorer(nusc)

    PATH = data_path + '/CAMFRONT.txt'

    with open(PATH) as f:
        image_token = [x.strip() for x in f.readlines()]

    annotations = []
    counter = 0
    # pdb.set_trace()
    for im_token in image_token:
        sample_data = nusc.get('sample_data', im_token)
        image_name = sample_data['filename']
        img = cv2.imread('/home/fengjia/data/sets/nuscenes/' + image_name)
        im = np.array(img)
        sample = nusc.get('sample', sample_data['sample_token'])
        lidar_token = sample['data']['LIDAR_TOP']

        # get ground truth boxes
        _, boxes, img_camera_intrinsic = nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ALL)

        for box in boxes:
            visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
            vis_level = int(nusc.get('visibility', visibility_token)['token'])
            if (vis_level != 3) and (vis_level != 4):
                continue
            ori_corners = view_points(box.corners(), view=np.array(img_camera_intrinsic, copy=True), normalize=True)
            if not(((ori_corners[0].max() - ori_corners[0].min()) > 64) and (
                    (ori_corners[1].max() - ori_corners[1].min()) > 64)):
                continue
            bottom_left = [np.int(ori_corners[0].min()), np.int(ori_corners[1].min())]
            top_right = [np.int(ori_corners[0].max()), np.int(ori_corners[1].max())]
            if not(box.name.split('.')[0] == 'vehicle'):
                continue
            # Find the crop area of the box
            width = ori_corners[0].max() - ori_corners[0].min()
            height = ori_corners[1].max() - ori_corners[1].min()
            x_mid = (ori_corners[0].max() + ori_corners[0].min()) / 2
            y_mid = (ori_corners[1].max() + ori_corners[1].min()) / 2
            side = max(width, height) * random.uniform(1.0, 1.2)

            if (x_mid - side / 2) < 0:
                side = x_mid * 2
            if (y_mid - side / 2) < 0:
                side = y_mid * 2

            # Crop the image
            bottom_left = [int(x_mid - side / 2), int(y_mid - side / 2)]
            top_right = [int(x_mid + side / 2), int(y_mid + side / 2)]
            shifted_corners = ori_corners
            shifted_corners[0] = ori_corners[0] - bottom_left[0]
            shifted_corners[1] = ori_corners[1] - bottom_left[1]
            crop_img = im[bottom_left[1]:top_right[1], bottom_left[0]:top_right[0]]

            # Scale to same size
            scale = 128 / side
            scaled = cv2.resize(crop_img, (128, 128))
            crop_img = np.transpose(scaled, (2, 0, 1))
            crop_img = crop_img.astype(np.float32)
            crop_img /= 255

            # Get corresponding point cloud for the crop
            points, depth, im_ = explorer.map_pointcloud_to_image(lidar_token, im_token)

            u, v = im.shape[:2]
            dep = np.zeros((u, v))
            for i in range(points.shape[1]):
                if points[1, i] > bottom_left[1] and points[1, i] < top_right[1]-1 and points[0, i] > bottom_left[0] and points[0, i] < top_right[0]-1:
                    dep[int(points[1, i]-bottom_left[1]), int(points[0, i]-bottom_left[0])] = depth[i]
            if np.count_nonzero(dep) < 100:
                continue
            dep = cv2.resize(dep, (128, 128))

            np.save(os.path.join(save_path, 'img_{}'.format(counter)), im)
            np.save(os.path.join(save_path, 'dep_{}'.format(counter)), dep)
            np.save(os.path.join(save_path, 'originalGT_{}'.format(counter)), ori_corners)
            np.save(os.path.join(save_path, 'shiftedGT_{}'.format(counter)), shifted_corners)
            print('saving number {}'.format(counter))
            counter += 1
    print(counter)
