import torch
from torch.autograd import Variable
import torch.nn as nn
from logger import Logger
from MLP import Fusion as MLP_Dense
from dataloader import local_dataloader
import numpy as np

import os

logger = Logger('./logs/Dense_Resnet')

nusc_classes = ['__background__',
                'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
                'trailer', 'truck']

batch_size = 4
nusc_set = local_dataloader(batch_size, len(nusc_classes), training=False)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size=batch_size, shuffle=False)
nusc_iters_per_epoch = int(len(nusc_set) / batch_size)


# model = MLP_Dense(k = 1, feature_transform = False)
model = MLP_Dense()
weight_path = os.path.join(r'./trained_model/2020_01_09__1', 'Epoch:70_loss:0.29883947529089755')
print(os.path.isfile(weight_path))
model.load_state_dict(torch.load(weight_path))
model.cuda()

img = torch.FloatTensor(1).cuda()
dep = torch.FloatTensor(1).cuda()
originalGT = torch.FloatTensor(1).cuda()
shiftedGT = torch.FloatTensor(1).cuda()
offSet = torch.FloatTensor(1).cuda()
cameraMatrix = torch.FloatTensor(1).cuda()
cameraFrameBox = torch.FloatTensor(1).cuda()


img = Variable(img)
dep = Variable(dep)
originalGT = Variable(originalGT)
shiftedGT = Variable(shiftedGT)
offSet = Variable(offSet)
cameraMatrix = Variable(cameraMatrix)
cameraFrameBox = Variable(cameraFrameBox)


output_dir = r'/home/fengjia/data/sets/nuscenes_temp/vehicle'
if not os.path.exists(output_dir):
    print('path incorrect')
    exit(0)


nusc_iter = iter(nusc_dataloader)
counter = 0
for step in range(nusc_iters_per_epoch):
    data = next(nusc_iter)
    with torch.no_grad():
        img.resize_(data[0].size()).copy_(data[0])
        dep.resize_(data[1].size()).copy_(data[1])
        originalGT.resize_(data[2].size()).copy_(data[2])
        shiftedGT.resize_(data[3].size()).copy_(data[3])
        offSet.resize_(data[4].size()).copy_(data[4])
        cameraMatrix.resize_(data[5].size()).copy_(data[5])
        cameraFrameBox.resize_(data[6].size()).copy_(data[6])
        #name.resize_(data[4].size()).copy_(data[4])

        model = model.eval()
        pred_offset, scores = model(img, dep, offSet, cameraMatrix)
        pred_offset = pred_offset.cpu()
        #data = data.cpu()
        for i in range(batch_size):
            pred_offset_i = pred_offset[i, :, :]
            np.save(data[7][i].replace('img', 'predOffset'), pred_offset_i)
            #np.save(os.path.join(output_dir, 'predOffset_{}'.format(counter)), pred_offset_i)
            print('counter', counter)
            counter += 1