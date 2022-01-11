import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np

image_names = glob.glob('data/spaghetti/*')
print(image_names)
for im_name in image_names:
    img = cv2.imread(im_name, 1)
    dim = (1000, int(1000 * img.shape[0] / img.shape[1]))
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(11,11),0)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

    laplacian = ((laplacian / np.max(laplacian)) * 255).astype('uint8')
    sobelx = ((sobelx / np.max(sobelx)) * 255).astype('uint8')
    sobely = ((sobely / np.max(sobely)) * 255).astype('uint8')

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()