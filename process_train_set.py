import random
import imgaug as ia
import cv2 as cv
import numpy as np
import os
from dataset_gen import *
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


path_inp = "data/dataset_out/images"
path_out = "data/dataset_out_2/images"
folders = ["train", "val"]
for folder in folders:
    path_inp_full = os.path.join(path_inp, folder)
    path_out_full = os.path.join(path_out, folder)
    im_names = get_img_paths(path_inp_full)
    for im_name in im_names:
        img = cv.imread(os.path.join(path_inp_full, im_name), 1)
        img_proc = cv.Canny(img, 100, 200)

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(img)
        ax2.imshow(img_proc)
        plt.show()
        cv.imwrite(os.path.join(path_out_full, im_name), img_proc)
