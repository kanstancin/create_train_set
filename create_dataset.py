import os.path
import random as rnd
import matplotlib.pyplot as plt
from add_spag import SpagAdder
from add_infill import InfillAdder
from dataset_gen import *

# spag paths
# inp_path_spag = "/home/ubuntu/workspace/datasets/spag_blender_imgs_v2"
inp_path_spag = "/home/cstar/workspace/data/spag_blender_imgs"

# infill paths
inp_path_infill = "data/crop"

# masks, printers, output path
inp_path_mask = "data/mask/out"
# inp_path_printer = "/home/ubuntu/workspace/create_train_set/data/3d_printers"
raw_data_dir = 'dataset-G10-Z130-D500-0/'  #
raw_data_path = '/home/cstar/workspace/grid-data/preproc_data/'  #
inp_path_printer = os.path.join(raw_data_path, raw_data_dir, f'dataset-im-diff-avg-{3}')
# inp_path_printer = "/home/cstar/workspace/data/bckg_imgs"
out_path = os.path.join(raw_data_path, raw_data_dir, f'dataset-im-diff-spag-avg-{3}')
# if not os.path.exists(out_path):
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(out_path, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(out_path, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(out_path, 'masks/train'), exist_ok=True)
labels_path = os.path.join(out_path, 'labels')
imgs_path = os.path.join(out_path, 'images')
masks_path = os.path.join(out_path, 'masks')

# download dataset:
# aws s3 --no-sign-request sync s3://open-images-dataset/validation [target_dir/validation]
# load printer img
im_names_bckgs = get_img_paths(inp_path_printer)
infill = InfillAdder(inp_path_infill, inp_path_mask)
spag = SpagAdder(inp_path_spag, inp_path_mask)
for i, im_bckg_name in enumerate(im_names_bckgs):
    print("*"*30+"\n", i)
    state = "/train"

    im_bckg = cv.imread(inp_path_printer + "/" + im_bckg_name, 1)
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)

    # add infill
    try:
        # im_bckg = infill(im_bckg)
        # add spag
        im_bckg, mask = spag(im_bckg)
    except Exception as e:
        print("\nERROR\n", e)
        continue

    # im_bckg = apply_image_transf(im_bckg)

    box = get_box_from_mask(mask)
    # try:
    #     box = cv_box_to_yolo(box, mask.shape)
    # except: continue;
    box = cv_box_to_yolo(box, mask.shape)
    # plt.imshow(im_bckg)
    # plt.show()

    # create grayscale
    # im_bckg = cv.cvtColor(im_bckg, cv.COLOR_RGB2GRAY)
    # im_bckg = [im_bckg,im_bckg,im_bckg]
    # im_bckg = np.transpose(im_bckg,(1,2,0))

    im_out_name = im_bckg_name
    imgs_path_full = imgs_path + state + "/" + im_out_name
    # im_bckg = cv.GaussianBlur(im_bckg, (3, 3), cv.BORDER_DEFAULT)
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_RGB2BGR)
    cv.imwrite(imgs_path_full, im_bckg)

    mask_path_full = f"{masks_path}{state}/mask{im_out_name}"
    cv.imwrite(mask_path_full, mask)

    labels_path_full = labels_path + state
    save_label(labels_path_full, im_out_name, box)
    # plt.imshow(im_bckg)
    # plt.show()

