from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_gcps
import rioxarray as rxa

from rasterio.control import GroundControlPoint

def get_gcp(bound, label, image):
    """
    https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html#ground-control-points
    
    """

    assert 'x' in image.dims
    assert 'y' in image.dims

    lat, lon = b.split('/')
    if 'N' in lat:
        lat = lat.strip("'N").split('°')
        lat = int(lat[0]) + float(int(lat[1])/60)
    elif 'S' in lat:
        lat = lat.strip("'N").split('°')
        lat = -int(lat[0]) + float(int(lat[1])/60)
    # have to assume all west because some don't have W in them
    lon = lon.strip("W").strip("'").split('°')
    lon = int(lon[0]) + float(int(lon[1])/60)
    if label == 'ul':
        x, y = 0, 0
    elif label == 'ur':
        x, y = 0, image.x[-1].data.ravel()[0]
    elif label == 'll':
        x, y = image.y[-1].data.ravel()[0], 0
    elif label == 'lr':
        x, y = image.x[-1].data.ravel()[0], image.y[-1].data.ravel()[0]
    else:
        raise KeyError(f"Unrecognized label {label}")
    return GroundControlPoint(x, y, lon, lat)

bounds = {}
bounds['ul'], bounds['ur'], bounds['ll'], bounds['lr'] = image_meta[['lat/long NW', 'lat/long NE', 'lat/long SW', 'lat/long SE']].values.ravel()
for k, b in bounds.items():
    bounds[k] = get_gcp(b, k, da)