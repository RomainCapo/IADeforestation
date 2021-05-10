import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import wkt
from shapely.geometry.point import Point

def compute_mean_point(df):
    coord_x = []
    coord_y = []
    
    for p in df['geometry'].to_numpy():
        coord_x.append(p.x)
        coord_y.append(p.y)
        
    return (np.mean(coord_x), np.mean(coord_y))

def find_nearest_points(mean_point, df_points, nb_to_draw):
    dict_points = {}
    for p in df_points.iterrows():
        current_point = p[1]['geometry']
        dict_points[p[0]] = np.sqrt((mean_point[0] - current_point.x) ** 2 + (mean_point[1] - current_point.y) ** 2)
        
    sorted_dict = dict(sorted(dict_points.items(), key=lambda item: item[1]))
    return list(sorted_dict)[0:nb_to_draw]

def stratified_df(geo_train, geo_val, nb_labels, nb_fold=5):
    
    prop0 = nb_labels[0]/nb_fold
    prop1 = nb_labels[1]/nb_fold
    
    nb_train0 = len(geo_train[geo_train['label'] ==0])
    nb_val0 = len(geo_val[geo_val['label'] ==0])

    
    mean_train = compute_mean_point(geo_train)
    mean_val = compute_mean_point(geo_val)
    
    if np.abs(nb_val0 - prop0) < 5 :
        pass
    else:
        if nb_val0 < prop0:
            nb_to_draw = int(prop0 - nb_val0)
            
            index_to_switch = find_nearest_points(mean_val, geo_train[geo_train['label'] == 0], nb_to_draw)
            rows = geo_train.loc[index_to_switch]
            geo_train = geo_train.drop(index_to_switch, inplace=False, axis='index')
            
            geo_val = geo_val.append(rows, ignore_index=True)
        else :

            nb_to_draw = int(nb_val0 - prop0)
            
            index_to_switch = find_nearest_points(mean_train, geo_val[geo_val['label'] == 0], nb_to_draw)
            rows = geo_val.loc[index_to_switch]
            geo_val = geo_val.drop(index_to_switch, inplace=False, axis='index')
            
            geo_train = geo_train.append(rows, ignore_index=True)
            
    nb_train1 = len(geo_train[geo_train['label'] ==1])
    nb_val1 = len(geo_val[geo_val['label'] ==1])

    if np.abs(nb_val1 - prop1) < 5 :
        pass
    else:
        if nb_val1 < prop1:
            nb_to_draw = int(prop1 - nb_val1)
            
            index_to_switch = find_nearest_points(mean_val, geo_train[geo_train['label'] ==1], nb_to_draw)
            rows = geo_train.loc[index_to_switch]
            geo_train = geo_train.drop(index_to_switch, inplace=False, axis='index')
            
            geo_val = geo_val.append(rows, ignore_index=True)
        else :
            nb_to_draw = int(nb_val1 - prop1)
            
            index_to_switch = find_nearest_points(mean_train, geo_val[geo_val['label'] == 1], nb_to_draw)
            rows = geo_val.loc[index_to_switch]
            geo_val = geo_val.drop(index_to_switch, inplace=False, axis='index')
            
            geo_train = geo_train.append(rows, ignore_index=True)
                             
    return (geo_train, geo_val)

def display_cross_val_map(data_train, data_val, maps, title, xlim=[106,110], ylim=[10,16], figsize=(6,6)):
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
    
def display_cross_val_map_class(data_train, data_val, maps, title, xlim=[106,110], ylim=[10,16], figsize=(12,6)):
    fig, axes = plt.subplots(1,2,figsize=figsize)

    maps.plot(ax=axes[0],facecolor='Grey', edgecolor='k',alpha=0.5,linewidth=0.3)
    maps.plot(ax=axes[1],facecolor='Grey', edgecolor='k',alpha=0.5,linewidth=0.3)

    data_train[data_train['label'] == 2].plot(ax=axes[0], markersize=1,categorical=True, legend=True, c="tab:green")
    data_val[data_val['label'] == 2].plot(ax=axes[0], markersize=1,categorical=True, legend=True, c="tab:orange")
    
    data_train[data_train['label'] != 2].plot(ax=axes[1], markersize=1,categorical=True, legend=True, c="tab:pink")
    data_val[data_val['label'] != 2].plot(ax=axes[1], markersize=1,categorical=True, legend=True, c="tab:blue")

    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    axes[0].set_xlabel("Latitude")
    axes[0].set_ylabel("Longitude")
    
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)

    axes[1].set_xlabel("Latitude")
    axes[1].set_ylabel("Longitude")

    legend = axes[0].legend(["Train coffee", "Val coffee"])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    
    legend = axes[1].legend(["Train other", "Val other"])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]

    fig.suptitle(title)
    fig.tight_layout()