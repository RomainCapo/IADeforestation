import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import wkt
from shapely.geometry.point import Point

def display_cross_val_map(data_train, data_val, maps, title, xlim=[106,110], ylim=[10,16], figsize=(6,6)):
    """Display of the map the train and valididation points.

    Args:
        data_train (geopandas.GeoDataFrame): GeoDataFrame with one column contains coordinates of points.
        data_val (geopandas.GeoDataFrame): GeoDataFrame with one column contains coordinates of points.
        maps (geopandas.GeoDataFrame): GeoDataFrame that contain a map. 
        title (string): Title of the map plot
        xlim (list, optional): Latitude range to display. Defaults to [106,110].
        ylim (list, optional): Longitude range to display. Defaults to [10,16].
        figsize (tuple, optional): Size of the plot. Defaults to (6,6).
    """
    fig, ax = plt.subplots(figsize=figsize)

    maps.plot(ax=ax,facecolor='Grey', edgecolor='k',alpha=0.5,linewidth=0.3)

    data_train.plot(ax=ax, markersize=1,categorical=True, legend=True, c="tab:green")
    data_val.plot(ax=ax, markersize=1,categorical=True, legend=True, c="tab:orange")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    legend = ax.legend(["Train", "Test"])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30] 

    fig.suptitle(title)
    fig.tight_layout()
    
def display_cross_val_map_class(data_train, data_val, maps, title, column_name='label', legend1=['Train coffee', 'Val coffee'], legend2=['Train other', 'Val other'], xlim=[106,110], ylim=[10,16], figsize=(12,6)):
    """Display of the map the train and valididation points for each class (max 2). Use to display for the 10 folds methods.

    Args:
        data_train (geopandas.GeoDataFrame): GeoDataFrame with one column contains coordinates of points.
        data_val (geopandas.GeoDataFrame): GeoDataFrame with one column contains coordinates of points.
        maps (geopandas.GeoDataFrame): GeoDataFrame that contain a map. 
        title (string): Title of the map plot
        column_name (str, optional): Column which contains the label. Defaults to 'label'.
        legend1 (list, optional): Legend for class 0 map. Defaults to ['Train coffee', 'Val coffee'].
        legend2 (list, optional): Legend for class 1 map. Defaults to ['Train other', 'Val other'].
        xlim (list, optional): Latitude range to display. Defaults to [106,110].
        ylim (list, optional): Longitude range to display. Defaults to [10,16].
        figsize (tuple, optional): Size of the plot. Defaults to (12,6).
    """
    fig, axes = plt.subplots(1,2,figsize=figsize)

    maps.plot(ax=axes[0],facecolor='Grey', edgecolor='k',alpha=0.5,linewidth=0.3)
    maps.plot(ax=axes[1],facecolor='Grey', edgecolor='k',alpha=0.5,linewidth=0.3)

    data_train[data_train[column_name] == 0].plot(ax=axes[0], markersize=1,categorical=True, legend=True, c="tab:green")
    data_val[data_val[column_name] == 0].plot(ax=axes[0], markersize=1,categorical=True, legend=True, c="tab:orange")
    
    data_train[data_train[column_name] == 1].plot(ax=axes[1], markersize=1,categorical=True, legend=True, c="tab:pink")
    data_val[data_val[column_name] == 1].plot(ax=axes[1], markersize=1,categorical=True, legend=True, c="tab:blue")

    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Longitude")
    
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)

    axes[1].set_xlabel("Latitude")
    axes[1].set_ylabel("Longitude")

    legend = axes[0].legend(legend1)
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    
    legend = axes[1].legend(legend2)
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]

    fig.suptitle(title)