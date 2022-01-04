import random
import imgaug as ia
import cv2 as cv
import numpy as np
import os
from dataset_gen import *
from imgaug import augmenters as iaa


path_inp = "data/spaghetti"
path_out = "data/spaghetti_proc"
im_names = get_img_paths(path_inp)
for im_name in im_names:
    img = cv.imread(path_inp + "/" + im_name, 1)
    dim = (1000, int(1000 * img.shape[0]/img.shape[1]))
    # resize image
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    img = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
    path_out_full = path_out + "/" + im_name

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = [img, img, img]
    img = np.transpose(img, (1, 2, 0))
    cv.imwrite(path_out_full, img)

