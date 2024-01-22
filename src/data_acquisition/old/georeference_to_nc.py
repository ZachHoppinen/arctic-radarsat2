"""
Ended up just using gdalwarp....

 georferencing and conversion to netcdfs

- something like getting the ground control points from ds['spatial_ref'].attrs['gcps']
- get geotransform (GT) from .transform.from_gcps() from rio library
- assign transform to ds
- calcualte lats and longs using x * GT[0] + GT[1] + y * GT[2] + GT[3] but this isn't the actual equation
- calculate phase and magnitude and convert bands to real and imaginary.
- save

"""

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

from affine import Affine

from rasterio.control import GroundControlPoint
from rasterio import transform

DATA_DIR = Path('/data/nga/')
tifs = list(DATA_DIR.glob('*SLC/*HH.tif'))

tif = tifs[1]
da = xr.open_dataarray(tif)
platform, order_key, product_key, delivery_key, beam_mode, date, time, pol, processing_level = tif.parent.stem.split('_')
# meta = pd.read_excel(Path('/home/rdcrlzh1s/radarsat/data/RS2_collection.xlsx'), parse_dates=['Acq. Date'])
# image_meta = meta.loc[(meta['Beam'] == beam_mode) & (meta['Acq. Date'] == date)]
gcps = [GroundControlPoint(row = g['properties']['row'], col = g['properties']['col'], x = g['geometry']['coordinates'][0], y = g['geometry']['coordinates'][1]) for g in da['spatial_ref'].attrs['gcps']['features']]
GT = transform.from_gcps(gcps)

lon, lat = GT * (da.x, da.y)

result = 1j*da.isel(band = 0); result += da.isel(band = 1)
mag = 10*np.log10(np.abs(result))

ds = xr.merge([da.isel(band = 0).drop('band').rename('real'), da.isel(band = 1).drop('band').rename('imaginary'), mag.drop('band').rename('magnitude'), xr.apply_ufunc(np.angle, result).drop('band').rename('phase')])

ds = ds.assign_coords(lat = (["x", "y"], lat.data)) 
ds = ds.assign_coords(lon = (["x", "y"], lon.data)) 

from scipy.interpolate import griddata

xs =  np.linspace(ds.lon.min(), ds.lon.max(), ds.x.size//50) # ds.x.size
ys = np.linspace(ds.lat.min(), ds.lat.max(), ds.y.size//50) # ds.y.size
xg, yg = np.meshgrid(xs, ys)

n = 1000

points = np.array(list(zip(ds.lon.data.ravel()[::n], ds.lat.data.ravel()[::n])))
das = []
for var in ['magnitude']: # ds.data_vars
    print(var)
    data = griddata(points, ds[var].data.ravel()[::n], (xg, yg), method = 'cubic', fill_value = np.nan)
    das.append(xr.DataArray(data, dims= ['y','x'], coords = [ys, xs]).rename(var))
re_ds = xr.merge(das)

