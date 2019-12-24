import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def show_image(normalized_img):
    normalized_img = normalized_img.transpose((1, 2, 0))
    return normalized_img

def draw_box_matlab(ax, corners, color):
    for i in range(4):
        ax.plot([corners.T[i][0], corners.T[i + 4][0]],
		        [corners.T[i][1], corners.T[i + 4][1]],
		        color=color[2], linewidth=2)

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2)
            prev = corner

    draw_rect(corners.T[:4], color[0])
    draw_rect(corners.T[4:], color[1])

if __name__ == '__main__':
    # root = r'/home/fengjia/data/sets/nuscenes_local/vehicle'
    root = r'/home/fengjia/data/sets/nuscenes_local/vehicle'
    for i in range(20):
        i += 0
        print(i)
        normalized_img = np.load(os.path.join(root, 'img_{}.npy'.format(i)))
        normalized_img = show_image(normalized_img)
        _, ax = plt.subplots(1, 1)
        ax.imshow(normalized_img)
        shiftedGT = np.load(os.path.join(root, 'shiftedGT_{}.npy'.format(i)))
        originalGT = np.load(os.path.join(root, 'originalGT_{}.npy'.format(i)))
        print(originalGT)

        predOffset = np.load(os.path.join(root, 'predOffset_{}.npy'.format(i)))
        predOffset = predOffset.transpose((1, 0))

        c = np.array([1,0,0])
        color = (c, c, c)
        draw_box_matlab(ax, shiftedGT, color)

        c = np.array([0, 1, 0])
        color = (c, c, c)
        draw_box_matlab(ax, predOffset, color)
        # plt.show()

