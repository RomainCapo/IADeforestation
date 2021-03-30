import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio

def display_img(img):
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_image = np.moveaxis(norm_image, 0,-1)
    plt.imshow(norm_image[:,:,1:4])
    plt.figure()

def show_grid_img(list_path):
    _, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for i,ax in enumerate(axs):
        img = rasterio.open(list_path[i]).read()
        img = np.moveaxis(img, 0,-1)[:,:,1:4]

        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        ax.imshow(img)
    plt.show()