import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random as rnd
from dataset_gen import *

class SpagAdder():
    def __init__(self, inp_path_frg, inp_path_mask):
        self.inp_path_frg = inp_path_frg
        self.inp_path_mask = inp_path_mask
        self.im_names_frgs = get_img_paths(inp_path_frg)
        self.im_names_masks = get_img_paths(inp_path_mask)

        self.count_frgs = -1

    def __call__(self, im_bckg):
        self.count_frgs += 1
        if (self.count_frgs == len(self.im_names_frgs)):
            rnd.shuffle(self.im_names_frgs)
            self.count_frgs = 0

        # frg_num = rnd.randint(1, len(self.im_names_frgs))
        im_name_frg = self.im_names_frgs[self.count_frgs]
        im_frg = cv.imread(self.inp_path_frg + "/" + self.im_names_frgs[self.count_frgs], cv.IMREAD_UNCHANGED)
        im_frg = cv.cvtColor(im_frg, cv.COLOR_BGRA2RGBA)

        print(im_name_frg)
        mask_num = rnd.randint(1, len(self.im_names_masks))
        im_mask = cv.imread(self.inp_path_mask + "/mask" + str(mask_num) + ".png", 0)


        # im_frg = apply_spag_transf(im_frg)
        # resize

        im_frg = crop_frg(im_frg)
        bckg_shape = im_bckg.shape
        # im_bckg = resize_bckg(im_frg, im_bckg)
        im_frg = resize_frg(im_frg, im_bckg)
        print("shapes:", bckg_shape, im_bckg.shape, im_frg.shape)
        im_bckg, mask = overlay_img(im_frg, im_bckg)
        im_bckg, mask = resize_imgs(im_bckg, mask, bckg_shape)

        # im_frg = resize_frg(im_frg, im_bckg)
        # threshold
        # im_frg = remove_bckg(im_frg)
        # im_bckg, mask = overlay_img(im_frg, im_bckg)

        # fig, axes = plt.subplots(2)
        # axes[0].imshow(im_frg)
        # axes[1].imshow(im_bckg)
        # plt.show()
        return im_bckg, mask

# inp_path_frg = "/home/cstar/Downloads/yandex.com/imgs1"
# inp_path_mask = "/home/cstar/Downloads/infill/input/mask/out"
# inp_path_printer = "/home/cstar/Downloads/3d_printer"
# out_path_overlayed = "/home/cstar/Downloads/yandex.com/imgs1_overlayed"
#
# # load printer img
# im_names_bckgs = get_img_paths(inp_path_printer)
# bckg_num = rnd.randint(1, len(im_names_bckgs))
# im_bckg = cv.imread(inp_path_printer + "/3d-printer" + str(bckg_num) + ".jpg", 1)
# im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)
#
# spag = SpagAdder(inp_path_frg, inp_path_mask)
# im_bckg, mask = spag(im_bckg)
#
# # im_name_bckg_no_ext, _ = os.path.splitext(im_name_bckg)
# # im_name_frg_no_ext, _ = os.path.splitext(im_name_frg)
# # out_file_path = out_path_overlayed+"/"+im_name_bckg_no_ext+"+"+im_name_frg_no_ext
# #
# # cv.imwrite(out_file_path+".jpg", im_bckg)
# # cv.imwrite(out_file_path+"_mask"+".jpg", mask)
# # cv.imshow("wind", mask)
# # cv.waitKey(5000)
#
# plt.imshow(mask)
# plt.show()
