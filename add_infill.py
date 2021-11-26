import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imgaug as ia
from imgaug import augmenters as iaa
from dataset_gen import *
import random as rnd
#ia.seed(4)

class InfillAdder:
    def __init__(self, inp_path_frg, inp_path_mask):
        self.inp_path_frg = inp_path_frg
        self.inp_path_mask = inp_path_mask
        self.im_names_frgs = get_img_paths(inp_path_frg)
        self.im_names_masks = get_img_paths(inp_path_mask)

        self.count_frgs = -1

    def __call__(self, im_bckg):
        # shuffle on overflow
        self.count_frgs += 1
        if (self.count_frgs == len(self.im_names_frgs)):
            rnd.shuffle(self.im_names_frgs)
            self.count_frgs = 0
        frg_num = rnd.randint(1,len(self.im_names_frgs))
        im_frg = cv.imread(self.inp_path_frg + "/" + self.im_names_frgs[self.count_frgs], cv.IMREAD_UNCHANGED)
        im_frg = cv.cvtColor(im_frg, cv.COLOR_BGRA2RGBA)

        mask_num = rnd.randint(1,len(self.im_names_masks))
        im_mask = cv.imread(self.inp_path_mask + "/mask" + str(mask_num) + ".png", 0)

        # apply mask
        im_mask = cv.resize(im_mask, tuple(np.flip(im_frg.shape[0:2])), interpolation=cv.INTER_NEAREST)
        im_frg[im_mask == 0] = [0, 0, 0, 0]
        # transforms
        print(im_frg.shape)
        im_frg = apply_infill_transf(im_frg)

        im_frg = resize_frg(im_frg, im_bckg)
        im_bckg, mask = overlay_img(im_frg, im_bckg)
        return im_bckg

# inp_path_frg = "/home/cstar/Downloads/infill/input/crop"
# inp_path_printer = "/home/cstar/Downloads/3d_printer"
# inp_path_mask = "/home/cstar/Downloads/infill/input/mask/out"
# out_path_overlayed = "/home/cstar/Downloads/infill/overlayed"
#
# # load printer img
# im_names_bckgs = get_img_paths(inp_path_printer)
# bckg_num = rnd.randint(1, len(im_names_bckgs))
# im_bckg = cv.imread(inp_path_printer + "/3d-printer" + str(bckg_num) + ".jpg", 1)
# im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)
#
# infill = InfillAdder(inp_path_frg, inp_path_mask, inp_path_printer)
# im_bckg = infill(im_bckg)
#
# plt.imshow(im_bckg,"gray")
# plt.show()

