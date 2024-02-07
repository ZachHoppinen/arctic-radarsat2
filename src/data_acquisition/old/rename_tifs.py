from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa

import enlighten

from rasterio.enums import Resampling

DATA_DIR = Path('/data/nga/')
tifs = list(DATA_DIR.glob('*SLC/*georeferenced.tif'))
out_dir = Path('/data/nga/ncs')

meta = pd.read_excel(Path('/home/rdcrlzh1s/radarsat/data/RS2_collection.xlsx'), parse_dates=['Acq. Date'])
dss = {name:[] for name in meta['AOI '].unique()}

manager = enlighten.get_manager()
pbar = manager.counter(total = len(tifs), desc = 'Converting to Stack')

# for name in dss.keys():
    # full = []
for tif in tifs:
    platform, order_key, product_key, delivery_key, beam_mode, date, time, pol, processing_level = tif.parent.stem.split('_')
    image_meta = meta.loc[(meta['Beam'] == beam_mode) & (meta['Acq. Date'] == date)]
    site_name = image_meta['AOI '].iloc[0]
    acq_date = image_meta['Acq. Date'].dt.strftime('%Y-%m-%d').iloc[0]
    pol = image_meta['Polarity'].iloc[0]

    # if image_meta['AOI '].iloc[0] != name:
    #     continue
    print(image_meta['AOI '].iloc[0])
    print(image_meta['Acq. Date'].iloc[0])

    da = xr.open_dataarray(tif)#.isel(x = slice(0, -1, 1000), y = slice(0, -1, 1000))

    result = 1j*da.isel(band = 1); result += da.isel(band = 0)
    mag = 10*np.log10(np.abs(result)).drop('band').rename('magnitude')
    mag = mag.where(np.isfinite(mag))

    phase = xr.apply_ufunc(np.angle, result).drop('band').rename('phase')

    date_ds = xr.merge([da.isel(band = 0).drop('band').rename('real'), da.isel(band = 1).drop('band').rename('imaj'), mag, phase])

    date_ds = date_ds.expand_dims(time = [image_meta['Acq. Date'].iloc[0]])

    # if len(full) == 0:
    #     print(f"starting: {image_meta['AOI '].iloc[0]}")    
    # else:
    #     date_ds = date_ds.rio.reproject_match(full[0], nodata = np.nan, resampling=Resampling.bilinear)
    
    out_dir.joinpath(site_name).mkdir(exist_ok = True)
    date_ds.to_netcdf(out_dir.joinpath(site_name, f'{site_name}_{acq_date}_{pol}.nc'), encoding={'real': {'dtype': 'float', '_FillValue': np.nan}, 'imaj': {'dtype': 'float', '_FillValue': np.nan}})
    
    # full.append(date_ds)

    pbar.update()

    # ds = xr.concat(full, 'time')
    # ds.to_netcdf(out_dir.joinpath(f'{name}.nc'), encoding={'real': {'dtype': 'float', '_FillValue': np.nan}, 'imaj': {'dtype': 'float', '_FillValue': np.nan}})