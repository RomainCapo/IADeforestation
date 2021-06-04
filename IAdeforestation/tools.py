import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sn
import os

from keras.models import load_model

def display_img(img):
    """Display a 13 band image to RGB.

    Args:
        img (numpy.ndarray): 13 bands image to plot in RGB.
    """
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_image = np.moveaxis(norm_image, 0,-1)
    plt.imshow(norm_image[:,:,1:4])
    plt.figure()

def show_grid_img(list_path):
    """Plot list of image if 9x9 grid.

    Args:
        list_path (list): List of image path to plot.
    """
    _, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for i,ax in enumerate(axs):
        img = rasterio.open(list_path[i]).read()
        img = np.moveaxis(img, 0,-1)[:,:,1:4]

        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        ax.imshow(img)
    plt.show()
    
def plot_confusion_matrix(cm, class_names, title="Confusion matrix"):
    """Display the confusion matrix.

    Args:
        cm (np.ndarray): confusion matrix compute with function sklearn.metrics.confusion_matrix.
        class_names (list): List of class name.
        title (str, optional): Title of the confusion matrix. Defaults to "Confusion matrix".
    """
    fig, ax  = plt.subplots(figsize=(4,4))
    heatmap = sn.heatmap(cm, annot=True,fmt='g', cmap='Blues', ax=ax,cbar=False)
    
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    
    heatmap.set_xticklabels(class_names)
    heatmap.set_yticklabels(class_names)
    
def get_best_cross_val_model(model_paths, model_name, scores, custom_objects=None):
    """Return model with the best cross val accuracy scores.

    Args:
        model_paths (string): Path of folder that contains models.
        model_name (string): Name of the model.
        scores (list): List of tuple(loss, accuracy).
        custom_objects (dict, optional): Dict of custom objects to load with model.. Defaults to None.

    Returns:
        [type]: [description]
    """
    losses, accs = zip(*scores)
    index = np.argmax(accs)
    return load_model(os.path.join(model_paths, model_name + '_' + str(index) + '.h5'), custom_objects=custom_objects)
    
def plot_acc_loss(path, title, number_of_folds=5):
    """
    Display plot of accuracy and loss for cross validation fold.

    Args:
        model_paths (string): Path of folder that contains models.
        number_of_folds (int): Number of fold for cross validation.
        title (string): Name of the plot.
    """
    fig, axes = plt.subplots(number_of_folds,2, figsize=(14,20))
    axes = axes.flatten()

    i = 0

    for file in os.listdir(path):
        if(file.split('.')[1] == 'npy'):
            chart = np.load(os.path.join(path, file), allow_pickle=True).item()

            # summarize history for accuracy
            axes[i].plot(chart['accuracy'])
            axes[i].plot(chart['val_accuracy'])
            axes[i].set_title(f'Model accuracy - Fold {(i+1)//2+1}')
            axes[i].set_ylabel('accuracy')
            axes[i].set_xlabel('epoch')
            axes[i].legend(['train', 'val'], loc='upper left')

            i+=1

            # summarize history for loss
            axes[i].plot(chart['loss'])
            axes[i].plot(chart['val_loss'])
            axes[i].set_title(f'Model loss - Fold {(i//2)+1}')
            axes[i].set_ylabel('loss')
            axes[i].set_xlabel('epoch')
            axes[i].legend(['train', 'val'], loc='upper left')

            i+=1

    fig.tight_layout()
    fig.suptitle(title, fontsize=30, y=1.05)

