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
nusc_set = local_dataloader(batch_size, len(nusc_classes), training=True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size=batch_size, shuffle=True)
nusc_iters_per_epoch = int(len(nusc_set) / batch_size)

num_epochs = 100

# model = MLP_Dense(k = 1, feature_transform = False)
model = MLP_Dense()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

regressor = nn.SmoothL1Loss(reduction='none')

img = torch.FloatTensor(1).cuda()
dep = torch.FloatTensor(1).cuda()
originalGT = torch.FloatTensor(1).cuda()
shiftedGT = torch.FloatTensor(1).cuda()


img = Variable(img)
dep = Variable(dep)
originalGT = Variable(originalGT)
shiftedGT = Variable(shiftedGT)


date = '2019_12_08__2'

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

        # Unsupervised loss
        # loss = regressor(pred_offset, shiftedGT).mean(dim=(1, 2)).view(batch_size, -1) * scores - 0.1 * torch.log(scores)
        loss = regressor(pred_offset, shiftedGT).mean(dim=(1, 2)).view(batch_size, -1)

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
    torch.save(model.state_dict(), os.path.join(output_dir, 'Epoch:{}_loss:{}'.format(epoch, loss_epoch)))
    loss_epoch = 0
