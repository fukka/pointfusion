import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    # root = r'/home/fengjia/data/sets/nuscenes_local/vehicle'
    l = glob.glob(r'/home/fengjia/data/sets/nuscenes_local/vehicle/originalGT_*.npy')
    l.sort(key=lambda s: int(s.split('_')[-1].split('.')[0]))
    root = r'/home/fengjia/data/sets/nuscenes_local/vehicle'
    loss = []
    # for i in range(len(l)):
    for i in range(5000):
        f = l[i]
        shiftedGT = np.load(l[i])
        predOffset = np.load(l[i].replace('originalGT', 'predOffset'))
        predOffset = predOffset.transpose((1, 0))
        # print(predOffset)
        # print((shiftedGT[:2, :]-predOffset[:2, :]).mean())
        loss.append((np.abs(shiftedGT[:2, :]-predOffset[:2, :])).mean())
    plt.plot(loss)
    plt.show()

