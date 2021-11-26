import random
import imgaug as ia
import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('.')
from dataset_gen import *
from imgaug import augmenters as iaa


inp_path_spag = "data/imgs1"
im_names = get_img_paths(inp_path_spag)
im_bckg = cv.imread(inp_path_spag + "/" + im_names[10], 1)

im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)

# hue_sat = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
# im_frg = hue_sat(image=im_bckg)

hue = iaa.AddToHue((-50,50))
aug = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
im_bckg = remove_bckg(im_bckg)
im_bckg[im_bckg[:,:,3] < 10] = [0,0,0,0]
im_bckg = im_bckg[:,:,:3]
im_bckg = hue(image=im_bckg)
im_frg = aug(image=im_bckg)

f, axarr = plt.subplots(1,2)
axarr[0].imshow(im_bckg)
axarr[1].imshow(im_frg)
plt.show()