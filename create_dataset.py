import random as rnd
import matplotlib.pyplot as plt
import traceback
from add_spag import SpagAdder
from add_infill import InfillAdder
from dataset_gen import *

# spag paths
inp_path_spag = "/home/cstar/workspace/blender_spag_generation/pictures_cropped"
# inp_path_spag = "/home/cstar/workspace/data/spag_blender_imgs"

# infill paths
inp_path_infill = "data/crop"

# masks, printers, output path
inp_path_mask = "data/mask/out"
inp_path_printer = "/home/cstar/workspace/data/backgrounds"
# inp_path_printer = "/home/cstar/workspace/data/bckg_imgs"
labels_path = "data/dataset_out/labels"
imgs_path = "data/dataset_out/images"

# download dataset:
# aws s3 --no-sign-request sync s3://open-images-dataset/validation [target_dir/validation]
# load printer img
im_names_bckgs = get_img_paths(inp_path_printer)
infill = InfillAdder(inp_path_infill, inp_path_mask)
spag = SpagAdder(inp_path_spag, inp_path_mask)
for i, im_bckg_name in enumerate(im_names_bckgs):
    print("*"*30+"\n", i)
    #if (i < 21893): continue;
    if (i < 10000): state = "/train"
    else: state = "/val"
    if (i == 11000): break;

    im_bckg = cv.imread(inp_path_printer + "/" + im_bckg_name, 1)
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)

    # add infill
    try:
        # im_bckg = infill(im_bckg)
        # add spag
        im_bckg, mask = spag(im_bckg)
    except Exception:
        print("\nERROR\n")
        traceback.print_exc()
        continue

    im_bckg = apply_image_transf(im_bckg)

    box = get_box_from_mask(mask)
    # try:
    #     box = cv_box_to_yolo(box, mask.shape)
    # except: continue;
    box = cv_box_to_yolo(box, mask.shape)
    # plt.imshow(im_bckg)
    # plt.show()

    # create grayscale
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_RGB2GRAY)
    im_bckg = [im_bckg,im_bckg,im_bckg]
    im_bckg = np.transpose(im_bckg,(1,2,0))

    im_out_name = "img" + str(i) + ".jpg"
    imgs_path_full = imgs_path + state + "/" + im_out_name
    # im_bckg = cv.GaussianBlur(im_bckg, (3, 3), cv.BORDER_DEFAULT)

    cv.imwrite(imgs_path_full, im_bckg)

    labels_path_full = labels_path + state
    save_label(labels_path_full, im_out_name, box)
    '''
    plt.imshow(im_bckg)
    plt.show()
    '''    
