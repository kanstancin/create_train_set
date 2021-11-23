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
    mask = mask.astype("uint8")
    # mask = [mask,mask,mask]
    # mask = np.transpose(mask,(1,2,0))
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

def get_box_from_mask(mask):
    # dilate mask
    kernel = np.ones((15, 15), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=1)
    kernel = np.ones((55, 55), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # find best cont
    cnts, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # iterate over all the contours.
    max_len = 0
    for contour in cnts:
        if len(contour) > max_len:
            max_len = len(contour)
            longest_cont = contour
    box = cv.boundingRect(longest_cont)
    print(box)
    return box

def cv_box_to_yolo(box, im_shape):
    im_height, im_width = im_shape
    print("im shape : ", im_shape)
    [j_min, i_min, width, height] = box
    x_center = j_min +  width / 2
    y_center = i_min + height / 2
    # normalize
    x_center /= im_width
    width /= im_width
    y_center /= im_height
    height /= im_height
    yolo_box = [x_center, y_center, width, height]
    return yolo_box

def save_label(path, im_name, box):
    box = ' '.join([str(elem) for elem in box])
    line = "0 " + str(box)
    print(line)
    im_name = os.path.splitext(im_name)
    path = path + "/" + im_name[0] + ".txt"
    with open(path, 'w') as f:
        f.write(line)