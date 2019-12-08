import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def show_image(normalized_img):
    img = normalized_img * 255 / 1
    fig = plt.Figure()
    plt.imshow(img)
    fig.show()


if __name__ == '__main__':
    root = r''
    normalized_img = np.load(os.path.join(root, 'img_0.npy'))
    show_image(normalized_img)
