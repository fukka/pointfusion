import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2
from logger import Logger

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import MLP_Global as MLP_Global
from dataloader import nuscenes_dataloader
# from utils import ResNet50Bottom, sampler, render_box, render_pcl, visualize_result, IoU

import matplotlib.pyplot as plt
import numpy as np
import pdb

import os


if __name__ == '__main__':
    logger = Logger('./logs/4')

    nusc_classes = ['__background__',
                    'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
                    'trailer', 'truck']
    batch_size = 1
    # nusc_sampler_batch = sampler(400, 2)
    nusc_set = nuscenes_dataloader(batch_size, len(nusc_classes), training=True)
    nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size=batch_size, shuffle=True)
    nusc_iters_per_epoch = int(len(nusc_set) / batch_size)
    print(len(nusc_set), nusc_iters_per_epoch)

    num_epochs = 2

    model = MLP_Global()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    regressor = nn.SmoothL1Loss()
    classifier = nn.BCELoss()

    im = torch.FloatTensor(1)
    points = torch.FloatTensor(1)
    offset = torch.FloatTensor(1)
    m = torch.FloatTensor(1)
    rot_matrix = torch.FloatTensor(1)
    gt_corners = torch.FloatTensor(1)

    im = im.cuda()
    points = points.cuda()
    offset = offset.cuda()
    m = m.cuda()
    rot_matrix = rot_matrix.cuda()
    gt_corners = gt_corners.cuda()

    im = Variable(im)
    points = Variable(points)
    offset = Variable(offset)
    m = Variable(m)
    rot_matrix = Variable(rot_matrix)
    gt_corners = Variable(gt_corners)

    date = '2019_11_11__1'

    out_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = out_dir + '/trained_model/' + date
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    min_loss = 100

    for epoch in range(1, num_epochs + 1):
        nusc_iter = iter(nusc_dataloader)
        loss_temp = 0
        loss_epoch = 0
        model = model.train()

        for step in range(nusc_iters_per_epoch):
            data = next(nusc_iter)
            with torch.no_grad():
                im.resize_(data[0].size()).copy_(data[0])
                points.resize_(data[1].size()).copy_(data[1])
                offset.resize_(data[2].size()).copy_(data[2])
                m.resize_(data[3].size()).copy_(data[3])
                rot_matrix.resize_(data[4].size()).copy_(data[4])
                gt_corners.resize_(data[5].size()).copy_(data[5])

                im_np = im.cpu().numpy()
                print(im_np.shape)
                img = np.zeros((128, 128, 3))
                for u in range(128):
                    for v in range(128):
                        img[u, v, :] = im_np[0, :, u, v]

                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # exit()
                boxes, classes = model(im, points)
                print(gt_corners)
                print(boxes)
                print(classes)

        print("Loss for Epoch {} is {}".format(epoch, loss_epoch))
        loss_epoch = 0
