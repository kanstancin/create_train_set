import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset_gen import *

inp_path = "/home/cstar/Downloads/yandex.com/imgs1"
out_path = "/home/cstar/Downloads/yandex.com/imgs1_out"

all_inps = get_img_paths(inp_path,verbose=False)

for im_name in all_inps[0:]:
    img_orig = cv.imread(inp_path+"/"+im_name)

    img_orig = remove_frg(img_orig)
#     # convert all pixels to the color of the centroids
#     segmented_image = centers[labels.flatten()]
#     # reshape back to the original image dimension
#     segmented_image = segmented_image.reshape(hs.shape)
#     # show the image
    print(im_name)
#     cv.imshow("wind", hsv)
#     cv.waitKey(15000)
#     edges = cv.Canny(img,10,20)
    im_name_no_ext, file_extension = os.path.splitext(im_name)
    cv.imwrite(out_path+"/"+im_name_no_ext+".png", img_orig)