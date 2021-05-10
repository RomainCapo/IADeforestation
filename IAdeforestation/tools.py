import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sn
import os

from keras.models import load_model

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
    
def plot_confusion_matrix(cm, class_names, title="Confusion matrix"):
    fig, ax  = plt.subplots(figsize=(4,4))
    heatmap = sn.heatmap(cm, annot=True,fmt='g', cmap='Blues', ax=ax,cbar=False)
    
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    
    heatmap.set_xticklabels(class_names)
    heatmap.set_yticklabels(class_names)
    
def get_best_cross_val_model(model_paths, model_name, scores, custom_objects=None):
    losses, accs = zip(*scores)
    index = np.argmax(accs)
    return load_model(os.path.join(model_paths, model_name + '_' + str(index) + '.h5'), custom_objects=custom_objects)