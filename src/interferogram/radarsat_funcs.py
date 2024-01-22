#!/home/rdcrlzh1s/miniforge3/envs/gamma/bin/python

from pathlib import Path
import shlex
import subprocess
from io import StringIO 
import sys
import shutil
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import py_gamma as pg


def force_delete_directory(directory):
  """Force delete a directory.

  Args:
    directory: The directory to delete.
  """

  try:
    shutil.rmtree(directory)
  except OSError as e:
    print(e)

def get_width(par_fp: Path):
    """
    get width of gamma image from par file
    """
    par = pg.ParFile(str(par_fp))
    if 'range_samples' in par.par_keys:
        width = int(par.get_dict()['range_samples'][0])
    elif 'width' in par.par_keys:
        width = int(par.get_dict()['width'][0])
    elif 'range_samp_1' in par.par_keys:
        width = int(par.get_dict()['range_samp_1'][0])
    elif 'interferogram_width' in par.par_keys:
        width = int(par.get_dict()['interferogram_width'][0])
    return width

def parse_coords(coord, northern_hemisphere = True, western_hemisphere = True):
    """
    parse coordinates from radarsat2 to decimal degrees
    """
    
    lat, long = coord.split('/')
    lat = lat.strip("'N").strip("'S")
    long = long.strip("'W").strip("'E")

    lats = lat.split('°')
    lat = float(lats[0]) + float(lats[1])/60
    lat = lat if northern_hemisphere else -lat

    longs = long.split('°')
    long = float(longs[0]) + float(longs[1])/60
    long = -long if western_hemisphere else long

    return (long, lat)

def execute(cmd):
    """
    execute a command in the shell as a subprocess
    v1.0 29-Jan-2014 clw
    """
    # log.info('\n'+cmd+'\n')
    args2 = shlex.split(cmd)
    subprocess.call(args2)
    return 0

class Capturing(list):
    """
    Captures print output of pygamma for parsing.
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def create_png(data_fp, par_fp = None, image_dir = None):
    if not par_fp:
        par_fp = create_par_fp(data_fp)
        if not par_fp.exists(): raise ValueError()
    
    if not image_dir: image_dir = Path('./').resolve()

    arr= pg.read_image(data_fp, par = str(par_fp))

    if np.iscomplexobj(arr):
        arr = np.angle(arr)

    vmin, vmax = np.quantile(arr, [0.01, 0.99])

    plt.imshow(arr, aspect = 'auto', vmax = vmin, vmin = vmax)

    plt.colorbar()

    plt.savefig(image_dir.joinpath(data_fp.stem).with_suffix(data_fp.suffix + '.png'))

    plt.close('all')

def create_par_fp(fp):
    return fp.with_suffix(fp.suffix + '.par')

def parse_bounds(output):
    lats = re.split(r'\s+', [l for l in output if 'min. latitude' in l][0])
    assert len(lats) == 8
    s, n = float(lats[3]), float(lats[-1])
    longs = re.split(r'\s+', [l for l in output if 'min. longitude' in l][0])
    assert len(longs) == 8
    w, e = float(longs[3]), float(longs[-1])

    return n, s, e, w