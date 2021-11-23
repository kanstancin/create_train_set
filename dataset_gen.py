import random
import imgaug as ia
import cv2 as cv
import numpy as np
import os
from imgaug import augmenters as iaa

def apply_infill_transf(im_frg):
    pc_aff = iaa.PiecewiseAffine(scale=(0.005, 0.013))
    im_frg = pc_aff(image=im_frg)
    pr_tf = iaa.PerspectiveTransform(scale=(0.03,0.15), keep_size=False)
    im_frg = pr_tf(image=im_frg)
    rotate = iaa.Affine(rotate=(-90, 90), scale=(1, 1), shear=(-50, 50))
    im_frg = rotate(image=im_frg)
    return im_frg

# spaghetti
def remove_bckg(img_orig):
    img = img_orig
    kernel_size = 3
    kernel = (kernel_size, kernel_size)
    #img = cv.GaussianBlur(img_orig,kernel,cv.BORDER_DEFAULT)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = [img[:,:,0],img[:,:,1], img[:,:,2]]
#     cv.imshow("wind", s)
#     cv.waitKey(15000)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    hs = img[:,:,1:2]
    pixel_values = hs.reshape((-1, 1))
    # convert to float
    pixel_values = np.float32(pixel_values)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    bcg_center_arg = np.argmax(centers,axis=0)[0]
    # flatten the labels array
    labels = labels.flatten()
    labels = labels.reshape(img.shape[0:2])
    img_orig = cv.cvtColor(img_orig, cv.COLOR_RGB2RGBA)
    img_orig[labels!=bcg_center_arg,3] = 0
    white_pxls = hsv[:,:,2] > 230
    #img_orig[white_pxls,3] = 0 # [0,0,0]
    # First create the image with alpha channel

    # Then assign the mask to the last channel of the image
    hsv[labels!=bcg_center_arg,:] = [0,0,0]
    return img_orig

def overlay_img(im_frg, im_bckg):
    frg_width = im_frg.shape[1]
    frg_height = im_frg.shape[0]
    bckg_width = im_bckg.shape[1]
    bckg_height = im_bckg.shape[0]
    i_over = int((bckg_height - frg_height) * random.uniform(0.0,1.0))
    j_over = int((bckg_width - frg_width) * random.uniform(0.0, 1.0))
    overlay_coord = (i_over,j_over)
    mask = np.zeros(im_bckg.shape[:2])
    mask[overlay_coord[0]:overlay_coord[0]+frg_height, \
                                    overlay_coord[1]:overlay_coord[1] + frg_width] = im_frg[:,:,3] >10
    im_bckg[mask!=False] = im_frg[im_frg[:,:,3] >10,:3]
    mask *= 255
    mask = [mask,mask,mask]
    mask = np.transpose(mask,(1,2,0))
    return im_bckg, mask

def resize_frg(im_frg, im_bckg):
    scale_fct = random.uniform(0.15, 0.35)# 0.1  # percent of original size
    width = int(im_bckg.shape[1] * scale_fct)
    height = int(im_bckg.shape[1] * scale_fct / im_frg.shape[1] * im_frg.shape[0])
    dim = (width, height)
    # resize image
    im_frg = cv.resize(im_frg, dim, interpolation = cv.INTER_AREA)
    return im_frg

# mask processing
def is_inp(img_name):
    return img_name.endswith(("jpg","jpeg","png"))

def get_img_paths(im_in_dir, verbose=True):
    all_inps = os.listdir(im_in_dir)
    all_inps = [name for name in all_inps if is_inp(name)]
    if verbose:
        print('*'*33)
        print("Found the following {} images:".format(len(all_inps)))
        for im_path in all_inps:
            print(im_path)
        print('*'*33)
    return all_inps

def process_mask(img):
    kernel = np.ones((10,10),np.uint8)
    img = cv.erode(img,kernel,iterations = 5)
    img = cv.dilate(img,kernel,iterations = 4)
    return img