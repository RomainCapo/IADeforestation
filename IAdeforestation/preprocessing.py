import rasterio
import cv2
import shapefile
import numpy as np
import os
import affine

from pyproj import Transformer

def normalize(image, mean, std):
    """Apply Z-normalisation to images based on eurosat images mean and std.
    Required Shape (width*height*channel)

    Args:
        image (numpy.array): source image
        mean (list): List of Eurosat dataset mean for each band
        std (list): List of Eurosat dataset std for each band

    Returns:
        numpy.array: Normalised image
    """
    for band_index in range(len(mean)):
        image[...,band_index] -= mean[band_index]
        image[...,band_index] /= std[band_index]
    return image

def get_raster_paths(base_path):
    """Return a dict with all raster (jp2 file). 

    Args:
        base_path (string): folder that contains Sentinel2 raster images

    Returns:
        dict: Dict with rastsers path. Key : band name, Value : raster path 
    """
    raster_paths = {}
    
    current_path = base_path
    safe_folder = os.listdir(current_path)[0]  
    current_path = os.path.join(current_path, safe_folder, 'GRANULE')

    granule_path = os.listdir(current_path)[0]
    current_path = os.path.join(current_path, granule_path, 'IMG_DATA')


    for p in os.listdir(current_path):
        if 'B' in p:
            band_name = p.split('_')[2].split('.')[0]
            raster_paths[band_name] =  os.path.join(current_path, p)
            
    return sorted(raster_paths.items())

def load_raster_img(raster_paths):
    """Load all rasters file in the given dict

    Args:
        raster_paths (dict): Dict with rastsers path. Key : band name, Value : raster path

    Returns:
        dict: Key : band name, Value : loaded raster
    """
    raster_dict = {}
    for band_name, path in raster_paths : 
        raster_dict[band_name] = rasterio.open(path, driver='JP2OpenJPEG')
        
    return raster_dict

def resample_bands(raster_dict, img_size=(10980,10980), interpolation=cv2.INTER_CUBIC):
    """Resample band to 10m resolution. Only bands that are not 10m resolution are resampled. Apply the given interpolation in param to image.

    Args:
        raster_dict (dict): key : band name, value : band as numpy array
        img_size (tuple, optional): source image size. Defaults to (10980,10980).
        interpolation (cv2.InterpolationFlags, optional): Interpolation applied to image. Defaults to cv2.INTER_CUBIC.

    Returns:
        dict: key: band name, value: resampled band if needed
    """
    image_dict = {}
    
    for k, i in raster_dict.items():
        tmp = i.read()[0]
        
        if tmp.shape != img_size:
            tmp = cv2.resize(tmp, img_size, interpolation=cv2.INTER_CUBIC)

        image_dict[k] = tmp
        
    return image_dict

def convert_to_format(coords, transformer):
    """Convert Sentinel2 coordinates to shapefile coordinate. Method from Bachelor work of J.Rod.

    Args:
        coords (tuple): tuple of Sentinel2 coord
        transformer (pyproj.transformer.Transformer): Transformer objet

    Returns:
        tuple: Tuple of new coordinates
    """
    conversion = transformer.transform(coords[0], coords[1])
    return (conversion[1], conversion[0])

def process_shapefile(shapefiles_paths):
    """Return coordinates lists from shapefile with corresponding label. Method from Bachelor work of J.Rod.

    Args:
        shapefiles_paths (list): List that contains Shapefile path

    Returns:
        List: list of shapefile coordinates. Element of list is tuple with first elem is shapefile coordinates, second elem corresponding label
    """
    points = []
    for path in shapefiles_paths:
        sf = shapefile.Reader(path)
        shapes = sf.shapes()
        for point in sf.records():
            points.append((shapes[point.oid].points[0],point[0]))
    return points

def get_labels(begin, end, points, transformer):
    """Return each label contained between given begin and end coordinates points. Method from Bachelor work of J.Rod.

    Args:
        begin (rasterio.transform): Begin point (Corresponds to the point at the top left)
        end (rasterio.transform): End point (Corresponds to the point below right)
        points (list): list of shapefile coordinates. Element of list is tuple with first elem is shapefile coordinates, second elem corresponding label
        transformer (pyproj.transformer.Transformer): Transformer objet

    Returns:
        set: Set of labels contained in the given point.
    """
    labels = set()
    long1, lat1 = convert_to_format(begin, transformer)
    long2, lat2 = convert_to_format(end, transformer)
    for point in points:
        if long1 <= point[0][0] <= long2 and lat2 <= point[0][1] <= lat1:
            labels.add(point[1])
    return labels

def split_export_img(original_img, raster_template, points, img_prefix='img', crop_size=64, split_size=171, shapefile_epsg=4326, export_folder='images'):
    """Splits the large image into split_size small image of size crop_size

    Args:
        original_img (np.array): Big image
        raster_template (rasterio): Rasterio base object. Each band can be given as an argument here
        points (List): list of shapefile coordinates. Element of list is tuple with first elem is shapefile coordinates, second elem corresponding label
        img_prefix (str, optional): Prefix which will be in front of the name of the exported image. Defaults to 'img'.
        crop_size (int, optional): Small image width and size. Defaults to 64.
        split_size (int, optional): Number of split in the big image. Defaults to 171.
        shapefile_epsg (int, optional): Shapefile EPSG. Defaults to 4326.
        export_folder (str, optional): Export folder. Defaults to 'images'.
    """
    NB_CHANNEL = original_img.shape[0]
    TOTAL_WIDTH = original_img.shape[1]
    TOTAL_HEIGHT = original_img.shape[2]
    
    orignal_transform = raster_template.transform
    orignal_crs = raster_template.crs
    orignal_dtypes = raster_template.dtypes[0]
    
    transformer = Transformer.from_crs(raster_template.gcps[1], shapefile_epsg)
    list_lab = []

    for i in range(split_size):
        for j in range(split_size):
            img = original_img[:,i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size]

            begin = orignal_transform * (j*crop_size,i*crop_size)
            end = orignal_transform * ((j+1)*crop_size, (i+1)*crop_size)

            new_transform = affine.Affine(10.0, 0.0, begin[0], 0.0, -10.0, begin[1])
                        
            labels = get_labels(begin, end, points, transformer)
            export_img(img, img_prefix + '_' + str(i) + '_' + str(j), labels, orignal_dtypes, orignal_crs, new_transform, export_folder=export_folder) 
            
def export_img(img, name, labels, dtype, crs, transform, export_folder='images'):
    """Save the raster image as tiff file on the disk

    Args:
        img (np.array): Image to save
        name (str): Image name
        labels (set): Set which contains the labels corresponding to the image
        dtype (rasterio.dtypes): dtype of source image
        crs (rasterio.crs): CRS of source image
        transform (rasterio.transform): Transform of source image
        export_folder (str, optional): Export folder. Defaults to 'images'.
    """
    target_label = 0
    if len(labels) == 1:
        target_label = list(labels)[0]
    elif len(labels) > 1:
        target_label = -1
    
    if target_label >= -1 and target_label <= 32: 
        with rasterio.open(os.path.join(export_folder, str(target_label), name+'.tiff'), 'w', driver='Gtiff',
                                                  width=img.shape[1],
                                                  height=img.shape[2],
                                                  count=img.shape[0],
                                                    dtype=dtype,
                                                   crs=crs,
                                                   transform=transform) as dst :
            dst.write(img)