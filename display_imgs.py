import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import random
import cv2
from glob import glob
from tqdm import tqdm

files = glob("../blender_spag_generation/pictures/*") #"data/detection_out/*"
for _ in range(1):
    row = 3
    col = 3
    grid_files = random.sample(files, row*col)
    images     = []
    for image_path in tqdm(grid_files):
        img          = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        dim = (500, int(500 * img.shape[0] / img.shape[1]))
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        images.append(img)
    fig, axes = plt.subplots(col, row)
    for i, img in enumerate(images):
        axes[i//col, i%col].imshow(img)
        axes[i // col, i % col].axis('off')
    fig.tight_layout()
    plt.savefig("blender_spag.png",dpi=500)
    plt.show()