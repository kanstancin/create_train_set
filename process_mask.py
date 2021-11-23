import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imgaug as ia
from imgaug import augmenters as iaa
from dataset_gen import *


inp_path = "/home/cstar/Downloads/infill/input/mask/in"
out_path = "/home/cstar/Downloads/infill/input/mask/out"

inp_names = get_img_paths(inp_path, verbose=True)

for im_name in inp_names:
    img_in = cv.imread(inp_path+"/"+im_name,0)
    img_in[img_in>150] = 255
    img_in[img_in<=150] = 0
    img_in = process_mask(img_in)
    cv.imwrite(out_path+"/"+im_name, img_in)
# plt.imshow(img_in,"gray")
# plt.show()