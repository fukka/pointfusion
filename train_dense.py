import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import MLP_Dense as MLP_Dense
from dataloader import local_dataloader
from utils import ResNet50Bottom, sampler, render_box, render_pcl, visualize_result

import matplotlib.pyplot as plt
import numpy as np
import pdb

import os

logger = Logger('./logs/Dense_Resnet')

nusc_classes = ['__background__',
                'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
                'trailer', 'truck']

batch_size = 4
nusc_set = local_dataloader(batch_size, len(nusc_classes), training=True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size=batch_size, shuffle=True)
nusc_iters_per_epoch = int(len(nusc_set) / batch_size)

num_epochs = 50

# model = MLP_Dense(k = 1, feature_transform = False)
model = MLP_Dense()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

regressor = nn.SmoothL1Loss(reduction='none')

img = np.load(self.img_list[index])
        dep = np.load(self.dep_list[index])
        originalGT = np.load(self.originalGT_list[index])
        shiftedGT = np.load(self.shiftedGT_list[index])
img = torch.FloatTensor(1).cuda()
dep = torch.FloatTensor(1).cuda()
originalGT = torch.FloatTensor(1).cuda()
shiftedGT = torch.FloatTensor(1).cuda()


img = Variable(img)
dep = Variable(dep)
originalGT = Variable(originalGT)
shiftedGT = Variable(shiftedGT)


date = '08_28_2019__2'

out_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = out_dir + '/trained_model/' + date
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for epoch in range(1, num_epochs + 1):
    scheduler.step()
    nusc_iter = iter(nusc_dataloader)
    loss_temp = 0
    loss_epoch = 0
    for step in range(nusc_iters_per_epoch):
        data = next(nusc_iter)
        with torch.no_grad():
            img.resize_(data[0].size()).copy_(data[0])
            dep.resize_(data[1].size()).copy_(data[1])
            originalGT.resize_(data[2].size()).copy_(data[2])
            shiftedGT.resize_(data[3].size()).copy_(data[3])

        optimizer.zero_grad()
        model = model.train()
        pred_offset, scores = model(img, dep)

        loss = 0
        n = 400

        # Unsupervised loss
        loss = regressor(pred_offset, shiftedGT).mean(dim=(2, 3)) * scores - 0.1 * torch.log(scores)
        loss = loss.sum(dim=1) / n
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
    loss_epoch /= nusc_iters_per_epoch
    logger.scalar_summary('loss', loss_epoch, epoch)

    print("Loss for Epoch {} is {}".format(epoch, loss_epoch))
    loss_epoch = 0
