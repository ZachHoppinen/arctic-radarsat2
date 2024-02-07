#!/home/ubuntu/miniconda3/envs/gamma/bin/python

import os
import sys
from pathlib import Path
import logging
import re

import pandas as pd
import py_gamma as pg

from radarsat_funcs import force_delete_directory, create_png, parse_coords, create_par_fp,\
    Capturing, parse_bounds, execute, get_width

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(logging.Formatter("%(name)s: %(message)s"))
logger.addHandler(stdout)

# tell GAMMA we don't want to run functions interactively
interactive = 0
overwrite = False

meta = pd.read_excel(Path('/home/ubuntu/arctic-radarsat2/data/RS2_collection.xlsx'), parse_dates=['Acq. Date'])

in_dir = Path("/data/nga/")
dirs = in_dir.glob("RS2_*_SLC")

# iterate through directories and identify their AOI and print list of pairs

loc_imgs = {}

for d in dirs:
    platform, order_key, product_key, delivery_key, beam_mode, date, time, pol, processing_level = d.stem.split('_')
    image_meta = meta.loc[(meta['Beam'] == beam_mode) & (meta['Acq. Date'] == date)]
    
    # make sure we have 1 image per path and 
    assert len(image_meta) == 1

    meta.loc[image_meta.index, 'dir_fp'] = d

dems = {'Pituffik': Path('/data/nga/dem/Pituffik.tif'), 'Oliktok': Path('/data/nga/dem/Oliktok.tif'),\
     'Utqiagvik': Path('/data/nga/dem/Utqiagvik.tif'), 'Toolik': Path('/data/nga/dem/Toolik.tif')}

# select only pituffik images

for loc in meta['AOI '].unique():
    
    if loc != 'Pituffik': continue
    loc_fps = meta[meta['AOI '] == loc]

    loc_dir = in_dir.joinpath(loc)
    loc_dir.mkdir(exist_ok = True)

    for orbit in loc_fps.Beam.unique():
        fps = loc_fps[loc_fps['Beam'] == orbit]
        orbit_dir = loc_dir.joinpath(orbit)
        orbit_dir.joinpath(orbit)

        

""".

test_dir = Path("/data/test")
image_dir = test_dir.joinpath('images')
input_dir = test_dir.joinpath('inputs')
proc_dir = test_dir.joinpath('processing')
out_dir = test_dir.joinpath('outputs')

for d in [input_dir, image_dir, proc_dir, out_dir]:
    force_delete_directory(d)
    d.mkdir(exist_ok = True)
"""

dates = {}