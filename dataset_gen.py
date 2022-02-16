import random
import imgaug as ia
import cv2 as cv
import numpy as np
import os
from imgaug import augmenters as iaa


def apply_image_transf(img):
    noise = iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05 * 255), per_channel=0.5))
    img = noise(image=img)
    blur = iaa.Sometimes(0.1, iaa.GaussianBlur(0, 2))
    img = blur(image=img)
    CutOutGauss = iaa.Sometimes(0.1, iaa.Cutout(nb_iterations=(1, 4), size=0.10, fill_mode="gaussian",
                                                fill_per_channel=True))
    img = CutOutGauss(image=img)
    CutOutConst = iaa.Sometimes(0.1,
                                iaa.Cutout(nb_iterations=(0, 3), size=0.20, fill_mode="constant", fill_per_channel=True,
                                           squared=False))
    img = CutOutConst(image=img)
    return img


def apply_infill_transf(im_frg):
    pc_aff = iaa.PiecewiseAffine(scale=(0.005, 0.013))
    im_frg = pc_aff(image=im_frg)
    pr_tf = iaa.PerspectiveTransform(scale=(0.03, 0.15), keep_size=False)
    im_frg = pr_tf(image=im_frg)
    rotate = iaa.Affine(rotate=(-5, 5), scale=(1, 1), shear=(-1, 1))
    im_frg = rotate(image=im_frg)
    return im_frg


def color_match_transf(im_frg, im_bckg):

    # extract alpha channel
    alpha_ch = im_frg[:, :, 3].reshape((-1, im_frg.shape[1], 1))
    # find dark and light colors in background image
    dark, light = get_dark_light(im_bckg)
    # range of pixel values that aren't transparent
    # final color correction
    im_frg = color_correct(im_frg, dark, light)

    return np.concatenate((im_frg, alpha_ch), axis=2)


def apply_spag_transf(im_frg):
    # extrack alpha ch
    alpha_ch = im_frg[:, :, 3].reshape((-1, im_frg.shape[1], 1))
    im_frg = im_frg[:, :, :3]
    # do transforms without alpha ch
    hue = iaa.AddToHue((-50, 50))
    im_frg = hue(image=im_frg)
    # return alpha ch
    im_frg = np.concatenate((im_frg, alpha_ch), axis=2)

    # do transforms
    pc_aff = iaa.Sometimes(0.1, iaa.PiecewiseAffine(scale=(0.005, 0.033))) #scale=(0.005, 0.023)
    im_frg = pc_aff(image=im_frg)
    pr_tf = iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.03, 0.35), keep_size=False)) #scale=(0.03, 0.15)
    im_frg = pr_tf(image=im_frg)
    rotate = iaa.Sometimes(0.1, iaa.Affine(rotate=(-90, 90), scale=(1, 1), shear=(-15, 15)))
    im_frg = rotate(image=im_frg)
    return im_frg


# background
def apply_bckg_transf(im_frg):
    # extrack alpha ch
    # do transforms without alpha ch
    hue = iaa.AddToHue((-10, 10))
    im_frg = hue(image=im_frg)

    # do transforms
    pc_aff = iaa.Sometimes(0.05, iaa.PiecewiseAffine(scale=(0.005, 0.01))) #scale=(0.005, 0.023)
    im_frg = pc_aff(image=im_frg)
    pr_tf = iaa.Sometimes(0.05, iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=False)) #scale=(0.03, 0.15)
    im_frg = pr_tf(image=im_frg)
    rotate = iaa.Sometimes(0.05, iaa.Affine(rotate=(-90, 90), scale=(1, 1), shear=(-15, 15), mode="reflect"))
    im_frg = rotate(image=im_frg)
    return im_frg


# spaghetti
def remove_bckg(img_orig):
    img = img_orig
    kernel_size = 3
    kernel = (kernel_size, kernel_size)
    # img = cv.GaussianBlur(img_orig,kernel,cv.BORDER_DEFAULT)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
    #     cv.imshow("wind", s)
    #     cv.waitKey(15000)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    hs = img[:, :, 1:2]
    pixel_values = hs.reshape((-1, 1))
    # convert to float
    pixel_values = np.float32(pixel_values)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    bcg_center_arg = np.argmax(centers, axis=0)[0]
    # flatten the labels array
    labels = labels.flatten()
    labels = labels.reshape(img.shape[0:2])
    img_orig = cv.cvtColor(img_orig, cv.COLOR_RGB2RGBA)
    img_orig[labels != bcg_center_arg, 3] = 0
    white_pxls = hsv[:, :, 2] > 230
    # img_orig[white_pxls,3] = 0 # [0,0,0]
    # First create the image with alpha channel

    # Then assign the mask to the last channel of the image
    hsv[labels != bcg_center_arg, :] = [0, 0, 0]
    return img_orig


def overlay_img(im_frg, im_bckg):
    # if im_frg.shape[0] > im_bckg.shape[0]:
    #     im_frg = im_frg[:im_bckg.shape[0], :]

    # find overlay coordinate
    frg_width = im_frg.shape[1]
    frg_height = im_frg.shape[0]
    bckg_width = im_bckg.shape[1]
    bckg_height = im_bckg.shape[0]
    i_over = int((bckg_height - frg_height) * random.uniform(0.0, 1.0))
    j_over = int((bckg_width - frg_width) * random.uniform(0.0, 1.0))
    overlay_coord = (i_over, j_over)
    # prepare foreground for overlay, i.e. make same shape
    im_frg_full = np.zeros([im_bckg.shape[0], im_bckg.shape[1], 4])
    im_frg_full[overlay_coord[0]:overlay_coord[0] + frg_height,
    overlay_coord[1]:overlay_coord[1] + frg_width] = im_frg[:, :, :]
    # create mask
    alpha_mask = im_frg_full[:, :, 3]
    alpha_mask[alpha_mask > 190] = alpha_mask[alpha_mask > 190] * 2
    alpha_mask = alpha_mask / np.max(alpha_mask)
    alpha_mask = np.array([alpha_mask, alpha_mask, alpha_mask]).transpose(1, 2, 0)
    # display for debug
    # import matplotlib.pyplot as plt
    # plt.imshow(alpha_mask)
    # plt.show()
    # plt.imshow(im_frg)
    # plt.show()
    # overlay images
    im_bckg = (im_bckg * (1 - alpha_mask) + im_frg_full[:, :, :3] * alpha_mask).astype("uint8")
    # covert mask to binary format
    alpha_mask = (alpha_mask[:, :, 0] > 0.1) * 255
    alpha_mask = alpha_mask.astype("uint8")
    return im_bckg, alpha_mask


def resize_imgs(img, mask, shape):
    shape = (shape[1], shape[0])
    img = cv.resize(img, shape, interpolation=cv.INTER_AREA)
    mask = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST)
    return img, mask


def resize_frg(im_frg, im_bckg):
    scale_fct = random.uniform(0.2, 0.45)  # 0.1  # percent of original size
    width = int(im_bckg.shape[1] * scale_fct)
    height = int(im_bckg.shape[1] * scale_fct / im_frg.shape[1] * im_frg.shape[0])
    dim = (width, height)
    # resize image
    im_frg = cv.resize(im_frg, dim, interpolation=cv.INTER_AREA)
    return im_frg


def resize_bckg(im_frg, im_bckg):
    scale_fct = random.uniform(1.3, 6)  # 0.1  # percent of original size
    width = int(im_frg.shape[1] * scale_fct)
    height = int(im_frg.shape[1] * scale_fct / im_bckg.shape[1] * im_bckg.shape[0])

    upper_width_limit = 12000
    if (width > upper_width_limit) or (height > upper_width_limit):
        print("\nLIMIT\n")
        if width > height:
            width = upper_width_limit
            height = int((upper_width_limit / im_bckg.shape[1]) * im_bckg.shape[0])
        else:
            height = upper_width_limit
            width = int((upper_width_limit / im_bckg.shape[0]) * im_bckg.shape[1])
    dim = (width, height)
    # resize image
    im_bckg = cv.resize(im_bckg, dim, interpolation=cv.INTER_AREA)
    return im_bckg


def crop_frg(im):
    non_zero = im[:, :, 3].nonzero()
    i_min, i_max = [np.min(non_zero[0]), np.max(non_zero[0])]
    j_min, j_max = [np.min(non_zero[1]), np.max(non_zero[1])]
    print(i_min, i_max)
    print(j_min, j_max)
    print(non_zero)
    return im[i_min:i_max, j_min:j_max]


# mask processing
def is_inp(img_name):
    return img_name.endswith(("jpg", "jpeg", "png"))


def get_img_paths(im_in_dir, verbose=True):
    all_inps = os.listdir(im_in_dir)
    all_inps = [name for name in all_inps if is_inp(name)]
    if verbose:
        print('*' * 33)
        print("Found the following {} images:".format(len(all_inps)))
        for im_path in all_inps:
            print(im_path)
        print('*' * 33)
    return all_inps


def process_mask(img):
    kernel = np.ones((10, 10), np.uint8)
    img = cv.erode(img, kernel, iterations=5)
    img = cv.dilate(img, kernel, iterations=4)
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
    x_center = j_min + width / 2
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


def get_dark_light(im_bkg):
    '''
    Finds the average lightest and darkest color in an RGB image
    and returns 2 lists containing the RGB values of those colors

    :param im_bkg: background image
    :return: dark: average dark color, light: average light color
    '''
    tmp = im_bkg
    im_bkg = cv.cvtColor(im_bkg, cv.COLOR_RGB2HSV)

    dark = [np.mean(tmp[im_bkg[:, :, 2] < 0.2 * 255, ch]) for ch in range(3)]
    light = [np.mean(tmp[im_bkg[:, :, 2] > 0.8 * 255, ch]) for ch in range(3)]
    print(f'dark: {dark}')
    print(f'light: {light}')
    return dark, light


def color_correct(im_frg, dark, light):
    '''
    Color corrects an image based on pre-defined dark and light colors

    :param im_frg: foreground image to be modified
    :param dark: dark color to match to
    :param light: light color to match to
    :return: color matched image
    '''
    coeffs = [np.polyfit((dark[ch], light[ch]), (0, 255), 1) for ch in range(3)]

    return np.clip(np.array([im_frg[:, :, ch] * coeffs[ch][0] + coeffs[ch][1] for ch in range(3)]).transpose(1,2,0), 0, 255).astype("uint8")


def bounding_box(alpha):
    '''
    Finds the range of pixels for which the image should be color corrected for

    :param alpha: the alpha channel of the image
    :return: the min and max pixel element
    '''

    non_zero = alpha.nonzero()
    i_min, i_max = [np.min(non_zero[0]), np.max(non_zero[0])]
    j_min, j_max = [np.min(non_zero[1]), np.max(non_zero[1])]
    
    return alpha[i_min:i_max, j_min:j_max]

