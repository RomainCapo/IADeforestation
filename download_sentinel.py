"""Download Sentinel2 images corresponding to the map.geojson file.
Images are downloaded in the folder OUTPUT_FOLDER
"""

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import os
import zipfile

OUTPUT_FOLDER = 'D:\\NewData'

api = SentinelAPI('alpha_03', 'R9846om157', 'https://scihub.copernicus.eu/dhus')

footprint = geojson_to_wkt(read_geojson('map.geojson'))
products = api.query(footprint,
                     platformname='Sentinel-2',
                     date=("20190501", '20191030'),
                     producttype='S2MSI1C',
                     cloudcoverpercentage=(0, 20))

print(len(products))

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

api.download_all(products, OUTPUT_FOLDER)

'''
for path in os.listdir(OUTPUT_FOLDER):
    if os.path.splitext(path)[1] == '.zip':
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_FOLDER)
        
        os.remove(path)'''