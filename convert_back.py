
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

# from utils import get_pointcloud, modified_map_pointcloud_to_image

import cv2
def convert_back(shifted, offSet, cameraMatrix):
    shifted_toOriginal = shifted.clone()
    shifted_toOriginal[:, 0, :] = torch.mul(shifted[:, 0, :], (offSet[:, 2] - offSet[:, 0]).unsqueeze(1)) / 256 + offSet[:,0].unsqueeze(1)
    shifted_toOriginal[:, 1, :] = torch.mul(shifted[:, 1, :], (offSet[:, 2] - offSet[:, 0]).unsqueeze(1)) / 256 + offSet[:,1].unsqueeze(1)
    shifted_toCameraFrame = shifted_toOriginal.clone()
    shifted_toCameraFrame[:, 0, :] = shifted_toOriginal[:, 0, :] * shifted_toOriginal[:, 2, :]
    shifted_toCameraFrame[:, 1, :] = shifted_toOriginal[:, 1, :] * shifted_toOriginal[:, 2, :]
    return torch.bmm(torch.inverse(cameraMatrix), shifted_toCameraFrame)


def convert_back_single(shifted, offSet, cameraMatrix):
    shifted[0, :] = shifted[0, :] * (offSet[2] - offSet[0]) / 256 + offSet[0]
    shifted[1, :] = shifted[1, :] * (offSet[3] - offSet[1]) / 256 + offSet[1]
    nbr_points = shifted.shape[1]
    shifted[:2, :] = shifted[:2, :] * shifted[2:3, :].repeat(2, 0).reshape(2, nbr_points)
    return np.dot(np.linalg.inv(cameraMatrix), shifted)


if __name__ == '__main__':
    data_path = r"/home/fengjia/data/sets/nuscenes_temp/vehicle"
    counter = 1
    shiftedGT = np.load(os.path.join(data_path, 'shiftedGT_{}.npy'.format(counter)))
    offSet = np.load(os.path.join(data_path, 'offSet_{}.npy'.format(counter)))
    originalGT = np.load(os.path.join(data_path, 'originalGT_{}.npy'.format(counter)))
    cameraMatrix = np.load(os.path.join(data_path, 'cameraMatrix_{}.npy'.format(counter)))
    cameraFrameBox = np.load(os.path.join(data_path, 'cameraFrameBox_{}.npy'.format(counter)))
    print(cameraFrameBox)
    print(convert_back(shiftedGT, offSet, cameraMatrix))

