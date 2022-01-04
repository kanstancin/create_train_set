import random as rnd
import matplotlib.pyplot as plt
from add_spag import SpagAdder
from add_infill import InfillAdder
from dataset_gen import *

# spag paths
inp_path_spag = "/home/ubuntu/workspace/datasets/spag_blender_imgs_v2"
# inp_path_spag = "/home/cstar/workspace/data/spag_blender_imgs"

# infill paths
inp_path_infill = "data/crop"

# masks, printers, output path
inp_path_mask = "data/mask/out"
inp_path_printer = "/home/ubuntu/workspace/create_train_set/data/3d_printers"
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
    if (i < 800): state = "/train"
    else: state = "/val"
    if (i == 1000): break;
    im_bckg = cv.imread(inp_path_printer + "/" + im_bckg_name, 1)

    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)

    # add infill
    try:
        im_bckg = infill(im_bckg)
        # add spag
        im_bckg, mask = spag(im_bckg)
    except:
        print("\nERROR\n")
        continue

    # im_bckg = cv.GaussianBlur(im_bckg, (5, 5), cv.BORDER_DEFAULT)

    box = get_box_from_mask(mask)
    # try:
    #     box = cv_box_to_yolo(box, mask.shape)
    # except: continue;
    box = cv_box_to_yolo(box, mask.shape)
    # plt.imshow(im_bckg)
    # plt.show()
    print("b shape ", im_bckg.shape)
    plt.imshow(im_bckg)
    plt.show()
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_RGB2BGR)

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

