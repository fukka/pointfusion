import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import Fusion as MLP_Dense
from dataloader import local_dataloader
from utils import ResNet50Bottom, sampler, render_box, render_pcl, visualize_result

import matplotlib.pyplot as plt
import numpy as np
import pdb

import os
import glob

logger = Logger('./logs/Dense_Resnet')

nusc_classes = ['__background__',
                'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
                'trailer', 'truck']

batch_size = 4
nusc_set = local_dataloader(batch_size, len(nusc_classes), training=True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size=batch_size, shuffle=False)

# model = MLP_Dense(k = 1, feature_transform = False)
model = MLP_Dense()
weight_path = r''
model.load_state_dict(torch.load(weight_path))
model.cuda()

img = torch.FloatTensor(1).cuda()
dep = torch.FloatTensor(1).cuda()
originalGT = torch.FloatTensor(1).cuda()
shiftedGT = torch.FloatTensor(1).cuda()


img = Variable(img)
dep = Variable(dep)
originalGT = Variable(originalGT)
shiftedGT = Variable(shiftedGT)


date = '08_28_2019__2'

nusc_iter = iter(nusc_dataloader)
loss_temp = 0
loss_epoch = 0
data = next(nusc_iter)
while data != None:
    data = next(nusc_iter)
    with torch.no_grad():
        img.resize_(data[0].size()).copy_(data[0])
        dep.resize_(data[1].size()).copy_(data[1])
        originalGT.resize_(data[2].size()).copy_(data[2])
        shiftedGT.resize_(data[3].size()).copy_(data[3])
    model = model.eval()
    pred_offset, scores, img_, imgfeat = model(img, dep)


    loss = regressor(pred_offset, shiftedGT).mean(dim=(1, 2)).view(batch_size, -1) * scores - 0.1 * torch.log(scores)
    # loss = loss.sum(dim=1) / n
    loss = loss.sum(dim=0) / batch_size
    loss_temp += loss.item()
    loss_epoch += loss.item()

    loss.backward()
    optimizer.step()

    # Finding anchor point and predicted offset based on maximum score
    max_inds = scores.max(dim=1)[1].cpu().numpy()
    p_offset = np.zeros((4, 8, 3))
    anchor_points = np.zeros((4, 3))
    truth_boxes = np.zeros((4, 8, 3))
    for i in range(0, 4):
        p_offset[i] = pred_offset[i][max_inds[i]].cpu().detach().numpy()
        # anchor_points[i] = (points.cpu().numpy().transpose((0, 2, 1)))[i][max_inds[i]]
        truth_boxes[i] = shiftedGT[i].cpu().numpy()

    # visualize_result(p_offset, anchor_points, truth_boxes)
    if step % 10 == 0 and step != 0:
        loss_temp /= 10
        print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
              .format(epoch, num_epochs + 1, step, nusc_iters_per_epoch, loss_temp))
        loss_temp = 0



if __name__ == '__main__':
    data_path = r'/home/fengjia/data/sets/nuscenes_local'
    img_list = []
    dep_list = []
    originalGT_list = []
    shiftedGT_list = []

    img_list_temp = glob.glob(os.path.join(data_path, 'img_*'))
    for img in img_list_temp:
        if (not os.path.isfile(img.replace('img', 'dep')) or (not os.path.isfile(img.replace('img', 'originalGT'))) or (not os.path.isfile(img.replace('img', 'shiftedGT')))):
            print(img)
            continue
        img_list.append(img)
        dep_list.append(img.replace('img', 'dep'))
        originalGT_list.append(img.replace('img', 'originalGT'))
        shiftedGT_list.append(img.replace('img', 'shiftedGT'))

    for i in range(len(img_list)):
        img = np.load(img_list[i])
        dep = np.load(dep_list[i])
        originalGT = np.load(originalGT_list[i])
        shiftedGT = np.load(shiftedGT_list[i])
