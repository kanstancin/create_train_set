import random as rnd
import matplotlib.pyplot as plt
from add_spag import SpagAdder
from add_infill import InfillAdder
from dataset_gen import *

# spag paths
inp_path_spag = "data/imgs1"

# infill paths
inp_path_infill = "crop"

# masks, printers, output path
inp_path_mask = "mask/out"
inp_path_printer = "3d_printers"
out_path_dataset = "dataset_out"

# download dataset:
# aws s3 --no-sign-request sync s3://open-images-dataset/validation [target_dir/validation]
# load printer img
im_names_bckgs = get_img_paths(inp_path_printer)
infill = InfillAdder(inp_path_infill, inp_path_mask)
spag = SpagAdder(inp_path_spag, inp_path_mask)
for i, im_bckg_name in enumerate(im_names_bckgs):
    im_bckg = cv.imread(inp_path_printer + "/" + im_bckg_name, 1)
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_BGR2RGB)

    # add infill
    im_bckg = infill(im_bckg)

    # add spag
    im_bckg, mask = spag(im_bckg)
    im_bckg = cv.GaussianBlur(im_bckg, (5, 5), cv.BORDER_DEFAULT)
    plt.imshow(im_bckg)
    im_bckg = cv.cvtColor(im_bckg, cv.COLOR_RGB2BGR)

    im_out_name = "img" + str(i) + ".jpg"
    cv.imwrite(out_path_dataset + "/" + im_out_name, im_bckg)
    plt.show()
