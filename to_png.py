import cv2 as cv

from dataset_gen import *

inp_path_frg = "/home/cstar/Downloads/infill/input/crop"
paths = get_img_paths(inp_path_frg)

# for im_name in paths:
#     ext = im_name[-3:]
#     if ext != "png":
#         img = cv.imread(inp_path_frg + "/" + im_name,1)
#         save_path = inp_path_frg + "/" + im_name
#         save_path = save_path[:-3] + "png"
#         print(save_path)
#         cv.imwrite(save_path, img)

for im_name in paths:
    img = cv.imread(inp_path_frg + "/" + im_name,1)
    save_path = inp_path_frg + "/" + im_name
    print(save_path)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
    cv.imwrite(save_path, img)