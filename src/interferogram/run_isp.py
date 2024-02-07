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

dir1 = Path("/data/nga/RS2_OK149295_PK1356401_DK1321366_U16_20230926_214549_HH_SLC")
dir2 = Path("/data/nga/RS2_OK149295_PK1356403_DK1321368_U16_20231020_214550_HH_SLC")
# dir1 = Path("/data/nga/RS2_OK149295_PK1356402_DK1321367_U18_20231013_215000_HH_SLC")
# dir2 = Path("/data/nga/RS2_OK149295_PK1356400_DK1321365_U18_20230919_215000_HH_SLC")
# dir1 = Path("/data/nga/RS2_OK149290_PK1356357_DK1321325_SLA76_20230602_025909_HH_SLC")
# dir2 = Path("/data/nga/RS2_OK149290_PK1356371_DK1321339_SLA76_20230813_025906_HH_SLC")
dems = {'Pituffik': Path('/data/nga/dem/Pituffik.tif'), 'Oliktok': Path('/data/nga/dem/Oliktok.tif'),\
     'Utqiagvik': Path('/data/nga/dem/Utqiagvik.tif'), 'Toolik': Path('/data/nga/dem/Toolik.tif')}

test_dir = Path("/home/ubuntu/arctic-radarsat2/data/test/")
image_dir = test_dir.joinpath('images')
input_dir = test_dir.joinpath('inputs')
proc_dir = test_dir.joinpath('processing')
out_dir = test_dir.joinpath('outputs')

for d in [input_dir, image_dir, proc_dir, out_dir]:
    force_delete_directory(d)
    d.mkdir(exist_ok = True)

meta = pd.read_excel(Path('/home/ubuntu/arctic-radarsat2/data/RS2_collection.xlsx'), parse_dates=['Acq. Date'])
dates = {}

#<================================== Convert radarsat .tifs to Gamma SCOMPLEX images =========================================>
# set up filepaths
fps = {}

ref_site = None
for d in [dir1, dir2]:
    platform, order_key, product_key, delivery_key, beam_mode, date, time, pol, processing_level = d.stem.split('_')
    image_meta = meta.loc[(meta['Beam'] == beam_mode) & (meta['Acq. Date'] == date)]

    assert len(image_meta) == 1
    site_name = image_meta['AOI '].iloc[0]

    # set reference site to site_name if this is the first run
    ref_site = site_name if ref_site is None else ref_site

    assert ref_site == site_name, f"Reference scene location {ref_site} doesn't match secondary location {site_name}"
    acq_date = image_meta['Acq. Date'].dt.strftime("%Y-%m-%d").iloc[0]
    pol = image_meta['Polarity'].iloc[0]

    dates[acq_date] = {'dir': d, 'meta': image_meta, 'site_name':site_name}

    ### creates gamma data and par files from radarsat2 images ###
    # args: Path to product.xml, path to lutSigma.xml, path to imagery_{pol}.tif, polarization, output_par_path, output_data_path
    ## inputs ##
    # xml file path with metadata
    xml_fp = d.joinpath("product.xml")
    # lutSigma file path with calibration data
    lutS_fp = d.joinpath("lutSigma.xml")
    # image fp to file with data
    img_fp = d.joinpath(f"imagery_{pol}.tif")
    ## outputs ##
    # slc filepath
    f_slc_fp = input_dir.joinpath(f"{acq_date}.f.slc")
    # par file_path
    f_slc_par = create_par_fp(f_slc_fp)
    # input geotiff -> gamma slc/par
    if not f_slc_fp.exists() or not f_slc_par.exists() or overwrite:
        logger.info("pg.par_RSAT2_SLC(xml_fp, lutS_fp, img_fp, pol, f_slc_par, f_slc_fp)")
        logger.info(f"pg.par_RSAT2_SLC({xml_fp}, {lutS_fp}, {img_fp}, {pol}, {f_slc_par}, {f_slc_fp})")
        pg.par_RSAT2_SLC(xml_fp, lutS_fp, img_fp, pol, f_slc_par, f_slc_fp)

    ### convert from Fcomplex to Scomplex and optionally crop ###
    ## inputs ##
    # in_data_fp
    f_slc_fp
    # in_par_fp
    f_slc_par
    ## outputs ##
    # out_data_fp
    slc_fp = proc_dir.joinpath(f"{acq_date}.slc")
    # out_par_fp
    slc_par = create_par_fp(slc_fp)
    ## controls ##
    # type of data conversion, 2 (convert fcomplex -> scomplex)
    conversion_type = 2
    # scale factor for input SLC
    scale_factor = '-'
    # sample to start crop on, number of sample to take from start line, line to start on, number of line to take
    crop_args = ['-', '-','-','-']# [1000, 2000, 1000, 2000]

    # copy and convert to scomplex (optional crop)
    if not slc_fp.exists() or not slc_par.exists() or overwrite:
        logger.info("SLC_copy(f_slc_fp, f_slc_par, slc_fp, slc_par, conversion_type, scale_factor, *crop_args)")
        logger.info(f"SLC_copy({f_slc_fp}, {f_slc_par}, {slc_fp}, {slc_par}, {conversion_type}, {scale_factor}, *{crop_args})")
        pg.SLC_copy(f_slc_fp, f_slc_par, slc_fp, slc_par, conversion_type, scale_factor, *crop_args)

    # create_png(slc_fp, slc_par, image_dir)

#<================================== Geocode reference image to DEM =========================================>

ref_date = min(dates)
logger.info(f'Coregistering to reference date {ref_date}')
# get metadata for reference image
ref_meta, ref_dir, site_name = dates[ref_date]['meta'], dates[ref_date]['dir'], dates[ref_date]['site_name']

# get bounds to download DEM from
# bounds = ref_meta[['lat/long NW', 'lat/long NE', 'lat/long SW', 'lat/long SE']].values.ravel()
# bounds = [parse_coords(b) for b in bounds]
# s, n = min([lat for (long, lat) in bounds]), max([lat for (long, lat) in bounds])
# w, e = min([long for (long, lat) in bounds]), max([long for (long, lat) in bounds])

with Capturing() as output:
    pg.SLC_corners(slc_par)

n, s, e, w = parse_bounds(output)
logger.info(f'Using bounds: n: {n}, s: {s}, e: {e}, w: {w}')

# download dem using elevation
try:
    dem_tif = dems[site_name]
    logger.info(f"Found dem file at {dem_tif}")
except KeyError:
    dem_tif = input_dir.joinpath(site_name).with_suffix('.tif')
    logger.info(f"eio clip -o {dem_tif} --bounds {w} {s} {e} {n}")
    execute(f"eio clip -o {dem_tif} --bounds {w} {s} {e} {n}")
    if not dem_tif.exists():
        raise ValueError(f"Failed to find DEM: {dem_tif}")

### multilook and create intensity image for reference image ###
## inputs ##
# slc image and par
ref_slc, ref_slc_par = proc_dir.joinpath(f"{ref_date}.slc"), proc_dir.joinpath(f"{ref_date}.slc.par")
## outputs ##
# reference multi looked intensity image
rmli = proc_dir.joinpath(ref_date).with_suffix(".rmli")
rmli_par = create_par_fp(rmli)
## controls ##
# 5 in range, 10 in azimuth
multi_look = [5, 10]
# multi look
logger.info("pg.multi_look(ref_slc, ref_slc_par,rmli, rmli_par , *multi_look)")
logger.info(f"pg.multi_look({ref_slc}, {ref_slc_par}, {rmli}, {rmli_par} , *{multi_look})")
pg.multi_look(ref_slc, ref_slc_par,rmli, rmli_par , *multi_look)

create_png(rmli, rmli_par, image_dir)

# import DEM to gamma format
### create dem parameter file to match cropped and multi looked data ###
## inputs ##
# parameters file to match to for dem parameter file
rmli_par
## output ##
# dem parameter file that matches mulitlooked amplitude image
dem_par = proc_dir.joinpath(site_name).with_suffix('.dem_par')
## controls ##
# nominal altitude to calculate bounds of radar image [default = 0]
nominal_alt = '-'
# delta_y dem y lin spacing for new DEM_Par file default = [-2.777777777e-4 deg]
#  delta_x dem x sample spacing default = [-2.777777777e-4 deg]
deltas =['-', '-']
# epsg for dem
epsg = 4326

logger.info('pg.create_dem_par(dem_par, rmli_par, *deltas, epsg, interactive)')
logger.info(f"pg.create_dem_par({dem_par}, {rmli_par}, *{deltas}, {epsg}, {interactive})")
pg.create_dem_par(dem_par, rmli_par, nominal_alt,  *deltas, epsg, interactive)

# now we feed the parameter file we created into the dem import to resample our DEM
### DEM import to GAMMA format ###
## inputs ##
# in tif fp
dem_tif
# dem_parameter file (already exists from last step)
dem_par

# geoid file
geoid_dem = Path(os.getenv('DIFF_HOME')).joinpath('scripts', 'egm2008-5.dem')
geoid_par = Path(os.getenv('DIFF_HOME')).joinpath('scripts', 'egm2008-5.dem_par')

## outputs ##
# dem_data_fp - to save data to
dem_fp = proc_dir.joinpath(site_name).with_suffix('.dem')

## controls ##
# geotiff input type = 0
input_type = 0
# use param file or geotiff metadata (0 for using existing param file, 1 for geotiff metadata)
param_priority = 0

# in tif_fp, dem_data_fp, existing dem_par_fp, input type (0 for geotiff), used dem_param priority (0) or extract from geotiff (1), dem to geoid reproject
logger.info('pg.dem_import(dem_tif, dem_fp, dem_par, input_type, param_priority, geoid_dem, geoid_par)')
logger.info(f"pg.dem_import({dem_tif}, {dem_fp}, {dem_par}, {input_type}, {param_priority}, {geoid_dem}, {geoid_par})")
pg.dem_import(dem_tif, dem_fp, dem_par, input_type, param_priority, geoid_dem, geoid_par)

create_png(dem_fp, dem_par, image_dir)

# not using this dem trans because we should already be in EQA format from using pre-existing par file
# dem_path = proc_dir.joinpath('EQA')
# pg.dem_trans(proc_dir.joinpath(site_name).with_suffix('.dem_par'), proc_dir.joinpath(site_name).with_suffix('.dem'), dem_path.with_suffix('.dem_par'), dem_path.with_suffix('.dem'), 1, 1)

### calcualte lookup table between slant range and DEM, and also shadow, layover, inc angle, local resoltuion, off nadir angle products ###
## inputs ##
# mli/slc parameter file
rmli_par
# DEM parameter file
dem_par
# dem data file
dem_fp

## outputs ##
# segment files - we don't create these so '-'
segment_fps = ['-', '-']
# look up table ebtween slant range and DEM projection
lt = dem_fp.with_suffix(f'.{ref_date}.lt')
# lover/shadow map in DEM coordinates
ls_fp = dem_fp.with_suffix(f'.{ref_date}.ls_map')
#layover/shadow map in slant range coordinates (not created)
ls_slant = '-'
# incidence angle map
inc_fp = dem_fp.with_suffix(f'.{ref_date}.inc')
## options ##
# lat/northing oversampling factor
lat_ovs = 1
# long/easting oversampling factor
lon_ovs = 1

# create lookup table and DEM products
logger.info('pg.gc_map2(rmli_par, dem_par, dem_fp, *segment_fps, lt, lat_ovs, lon_ovs, ls_fp, ls_slant, inc_fp)')
logger.info(f"pg.gc_map2({rmli_par}, {dem_par}, {dem_fp}, *{segment_fps}, {lt}, {lat_ovs}, {lon_ovs}, {ls_fp}, {ls_slant}, {inc_fp})")
pg.gc_map2(rmli_par, dem_par, dem_fp, *segment_fps, lt, lat_ovs, lon_ovs, ls_fp, ls_slant, inc_fp)

create_png(inc_fp, dem_par, image_dir)

create_png(ls_fp, rmli_par, image_dir)
create_png(lt, dem_par, image_dir)

### Calculate radar cross sectional area for sigma0 and gamma0 calculations
# we are gonna use this simulated image to co-register our DEM to the reference amplitude image
## inputs ##
# mli par
rmli_par
# DEM par
dem_par
# dem mdata
dem_fp
# look up table (from last step)
lt
# layover shadow map in map geometry
ls_fp
# inc map
inc_fp
## outputs ##
# sigma naught pixel normalization area
pix_sigma0 = dem_fp.with_suffix(f'.{ref_date}.pix_sigma0')

# calculate sigma0 pixel normalization area
logger.info('pg.pixel_area(rmli_par, dem_par, dem_fp, lt, ls_fp, inc_fp, pix_sigma0)')
logger.info(f"pg.pixel_area({rmli_par}, {dem_par}, {dem_fp}, {lt}, {ls_fp}, {inc_fp}, {pix_sigma0})")
pg.pixel_area(rmli_par, dem_par, dem_fp, lt, ls_fp, inc_fp, pix_sigma0)

create_png(pix_sigma0, rmli_par, image_dir)

### create a new parameter file to the differential interferogram ###
## inputs
# rmli parameter file (reference image)
rmli_par
# second image file (None yet)
sec_par = '-'
## outputs ##
diff_par = proc_dir.joinpath(ref_date).with_suffix('.diff_par')
## controls
# par_type (1 for ISP inteferogram parameter file)
par_type = 1

logger.info('pg.create_diff_par(rmli_par, sec_par, diff_par, par_type, interactive)')
logger.info(f"pg.create_diff_par({rmli_par}, {sec_par}, {diff_par}, {par_type}, {interactive})")
pg.create_diff_par(rmli_par, sec_par, diff_par, par_type, interactive)

### Offset estimation between MLI images using intensity cross-correlation ###
# Offset estimation between MLI images using intensity cross-correlation of patches
# compares DEM area to amplitude image to match
## inputs
# intensity image 1
pix_sigma0
# intensity image 2
rmli
# diff par file
diff_par
## outputs ##
# offset estimates in range and azimuth (fcomplex)
offs = proc_dir.joinpath(ref_date).with_suffix('.offs')
# signal to noise ratio file - cross-correlation of each patch (0.0->1.0) (float)
snr = proc_dir.joinpath(ref_date).with_suffix('.snr')
# range patch size, azimuth patch size
patch_n = [256, 256]
# offset estimates in range and azimuth (text format)
offs_text = proc_dir.joinpath('offsets')
## controls ##
# n_ovr - mli image oversampling factor
mli_overs = 1
# number of offset esimates in (range, azimuth ) directions
offset_positions = (64, 64)
# threshold cross correlation threshold
cc_thres = 0.3
#Lanczos interpolator order 5 -> 9 (enter - for default: 5)
lancoz = '-'
# bandwidth fraction of low-pass filter on intensity data (0.0->1.0) (enter - for default: 0.8)
bw_frac = '-'
# print flag (default)
offset_print = '-'
# plotting flag (0 = none default, 1 = screen output, 2 = screen and png format plots, 3 = pdf outputs)
# 2 and 3 make a lot of plots...
offset_plotting = 0

logger.info('pg.offset_pwrm(pix_sigma0, rmli, diff_par, offs, snr, *patch_n, offs_text, mli_overs, *offset_positions, cc_thres, \
    lancoz, bw_frac, offset_print, offset_plotting)')
logger.info(f"pg.offset_pwrm({pix_sigma0}, {rmli}, {diff_par}, {offs}, {snr}, *{patch_n}, {offs_text}, {mli_overs}, *{offset_positions}, {cc_thres}, \
    {lancoz}, {bw_frac}, {offset_print}, {offset_plotting})")
pg.offset_pwrm(pix_sigma0, rmli, diff_par, offs, snr, *patch_n, offs_text, mli_overs, *offset_positions, cc_thres, \
    lancoz, bw_frac, offset_print, offset_plotting)

### Range and azimuth offset polynomial estimation using SVD ###
# inputs #
# offsets in azimuth and range
offs
# ccp - cross correlatin of each patch
snr
# diff parameter file
diff_par
## outputs ##
#  culled range and azimuht offset estimates
cull_offs = proc_dir.joinpath(ref_date).with_suffix('.coffs')
# culled offset estimates and cross correlation values
cull_offsets = proc_dir.joinpath(ref_date).with_suffix('.coffsets')
## controls
# cross correlation threshold
cc_thres = 0.4
# npoly - polynomila parameters default =4
npoly = 4

# create polynomial from offset calculation
logger.info('pg.offset_fitm(offs, snr, diff_par, cull_offs, cull_offs, cc_thres, npoly)')
logger.info(f"pg.offset_fitm({offs}, {snr}, {diff_par}, {cull_offs}, {cull_offsets}, {cc_thres}, {npoly})")
pg.offset_fitm(offs, snr, diff_par, cull_offs, cull_offsets, cc_thres, npoly)

### improve our look up table using offset parameters from above. ###
## inputs ##
# gc look up table
lt
# width of geocding lookup table (samples) - same as DEM
dem_width = get_width(dem_par)
# diff par
diff_par
## outputs ##
# improved geocoded look up table
lt_fine = dem_fp.with_suffix(f'.{ref_date}.lt_fine')
## controls ##
# reference image flag (offsets measured relative to ref image or simulared sar image)
# we are using a simulated sar images from the sigma0
reference_img_flag = 1

logger.info("pg.gc_map_fine(lt, dem_width, diff_par, lt_fine, reference_img_flag)")
logger.info("pg.gc_map_fine({lt}, {dem_width}, {diff_par}, {lt_fine}, {reference_img_flag})")
pg.gc_map_fine(lt, dem_width, diff_par, lt_fine, reference_img_flag)

create_png(lt_fine, dem_par, image_dir)

### re-run sigma0, gamma0 image with refined look up table ###
## inputs ##
# mli par
rmli_par
# DEM par
dem_par
# dem mdata
dem_fp
# refined look up table (from last step)
lt_fine
# layover shadow map in map geometry
ls_fp
# inc map
inc_fp
## outputs ##
# sigma naught pixel normalization area
pix_sigma0
# gamma naught pixel normalization area
pix_gamma0 = dem_fp.with_suffix(f'.{ref_date}.pix_gamma0')

# create sigma0, gamma0 again with refined lookup table
logger.info("pg.pixel_area(rmli_par, dem_par, dem_fp, lt_fine, ls_fp, inc_fp, pix_sigma0, pix_gamma0)")
logger.info(f"pg.pixel_area({rmli_par}, {dem_par}, {dem_fp}, {lt_fine}, {ls_fp}, {inc_fp}, {pix_sigma0}, {pix_gamma0})")
pg.pixel_area(rmli_par, dem_par, dem_fp, lt_fine, ls_fp, inc_fp, pix_sigma0, pix_gamma0)

create_png(pix_gamma0, rmli_par, image_dir)

### calculate lookup tables with coverage for layover (applying the refinement again) ###
# copy our dem par file to a temporary location before this step
dem_par_tmp = dem_par.with_suffix(dem_par.suffix + '.tmp')
execute(f"cp {dem_par} {dem_par_tmp}")

# Calculate terrain-geocoding lookup table and DEM derived data products
## inputs ##
# rmli par
rmli_par
# OFF_par - ISP offset/interferogram parameter file (enter - if geocoding SLC or MLI data)
off_par = None
# dem_par - dem  parameter file, dem - dem data file
dem_par, dem_fp
## outputs ##
# dem_segment par file - using temporary dem par file for this
dem_par_tmp 
# dem segment data file - not saved
dem_segment_fp = '-'
# lookup table
# geocoding lookup table
lt_tmp = dem_fp.with_suffix(f'.{ref_date}.lt.tmp')
## controls ##
# oversample factor in lat, lon
oversample_factor = [1, 1]
# simulated sar backscatter image in DEM geometry
sim_sar = dem_fp.with_suffix(f'.{ref_date}.sim_sar')
# zenith angle of surface normal
u = proc_dir.joinpath("u")
# orientation agle of n
v = proc_dir.joinpath("v")
# local incidence angle
inc = proc_dir.joinpath("inc")
# projection angle (between surface noraml and image plane)
psi = proc_dir.joinpath("psi")
# pixel normalization area
pix = proc_dir.joinpath("pix")
# layover shadow map
ls_map = proc_dir.joinpath("ls_map")
## controls ##
# frame - number of DEM pixels to aadd around area coverd by SAR iamges (default = \8)
frame = 8
# what to do with areas of layover/shadow ( 0 - set to 0,0), (1 - linear interp across,), (2 - actual value (default), (3 - nn thinned))
ls_mode = 3
# r_oversample - range over sampling factor for nn-thined radar shadow/layover default = 2
# tutorial uses 1024
r_oversample = 2.0

# create new products
logger.info("pg.gc_map1(rmli_par, off_par, dem_par, dem_fp, dem_par_tmp, dem_segment_fp, lt_tmp, *oversample_factor, sim_sar, u, v, inc, psi, pix, ls_map, frame, ls_mode, r_oversample )")
logger.info(f"pg.gc_map1({rmli_par}, {off_par}, {dem_par}, {dem_fp}, {dem_par_tmp}, {dem_segment_fp}, {lt_tmp}, *{oversample_factor}, {sim_sar}, {u}, {v}, {inc}, {psi}, {pix}, {ls_map}, {frame}, {ls_mode}, {r_oversample} )")
pg.gc_map1(rmli_par, off_par, dem_par, dem_fp, dem_par_tmp, dem_segment_fp, lt_tmp, *oversample_factor, sim_sar, u, v, inc, psi, pix, ls_map, frame, ls_mode, r_oversample )

### interpolate through nans regions ###
# Weighted interpolation of gaps in 2D data using an adaptive smoothing window
## inputs ##
# temporary lookup table (from last step) with gaps
lt_tmp
## outputs ##
# data with gaps filled
lt
## controls ##
# width of data
# for some reason I thought this would be the slant range width get_width(rmli_par)?
dem_width
# r_max - maximum interpolatioin window radio
r_max = 25
# min number of points to use[default = 16]
min_n = 5
# max number of points to use [default = 16]
max_n = 16
# weighting mode (0 - constant, 1 - IDW, 2 - inverse distance squared, 3 - exp)
w_mode = 2
# dtype = 0 (fcomplex)
dtype = 0
# cp_data (0 = do not copy, 1 = copy to output [default])
cp = 1

logger.info("pg.interp_ad(lt_tmp, lt, dem_width, r_max, min_n, max_n, w_mode, dtype, cp)")
logger.info(f"pg.interp_ad({lt_tmp}, {lt}, {dem_width}, {r_max}, {min_n}, {max_n}, {w_mode}, {dtype}, {cp})")
pg.interp_ad(lt_tmp, lt, dem_width, r_max, min_n, max_n, w_mode, dtype, cp)

# remove gapped temporary lookup table
execute(f"rm {lt_tmp}")

### again calculate refined lookup table with gap filled layover ###
## inputs ##
# gc look up table
lt
# width of geocding lookup table (samples) - same as DEM
dem_width
# diff par
diff_par
## outputs ##
# improved geocoded look up table
lt_fine
## controls ##
# reference image flag (offsets measured relative to ref image or simulared sar image)
# we are using a simulated sar images from the sigma0
reference_img_flag = 1
pg.gc_map_fine(lt, dem_width, diff_par, lt_fine, reference_img_flag)

# remove unrefined lookup table
execute(f"rm {lt}")

### transform reference backscatter image to map segment geometry ###
# Geocoding of image data using a geocoding lookup table
## inputs ##
# slant range data file
rmli
# width of data file
rmli_width = get_width(rmli_par)
# lookup table
lt_fine
## outputs ##
# data_out output data file
rmli_geo = proc_dir.joinpath(ref_date).with_suffix(".rmli.geo")
# out width
dem_width
# out lines ['-' for number of lines in gc_map]
out_lines = '-'
# interp mode  3 = bicubic-sqrt spline
interp_mode = 3
# dtype # 0 for float
dtype = 0 

logger.info("pg.geocode_back(rmli, rmli_width, lt_fine, rmli_geo, dem_width, out_lines, interp_mode, dtype)")
logger.info(f"pg.geocode_back({rmli}, {rmli_width}, {lt_fine}, {rmli_geo}, {dem_width}, {out_lines}, {interp_mode}, {dtype})")
pg.geocode_back(rmli, rmli_width, lt_fine, rmli_geo, dem_width, out_lines, interp_mode, dtype)

create_png(rmli_geo, dem_par, image_dir)

bmp = image_dir.joinpath(f'{ref_date}.rmli.geo.bmp')

pg.raspwr(rmli_geo, dem_width, 1, 0, 1, 1, 1., .35, 'gray.cm', bmp)
kml = image_dir.joinpath(f'{ref_date}.rmli.geo.kml')
pg.kml_map(bmp, dem_par, kml)

### transform DEM height into MLI geometry ###
# Forward geocoding transformation using a lookup table
## inputs ##
# lookup table
lt_fine
# input data file
dem_fp
# input_data width
dem_width
# output_data filepath
ref_dem = proc_dir.joinpath(ref_date).with_suffix('.hgt')
# output width
rmli_width
# out lines ['-' for number of lines in gc_map]
out_lines = int(pg.ParFile(str(rmli_par)).get_dict()['azimuth_lines'][0])
# interp mode  2 = SQR(1/dist)
interp_mode = 2
# dtype # 0 for float
dtype = 0 

logger.info("pg.geocode(lt_fine, dem_fp, dem_width, ref_dem, rmli_width, out_lines, interp_mode, dtype)")
logger.info(f"pg.geocode({lt_fine}, {dem_fp}, {dem_width}, {ref_dem}, {rmli_width}, {out_lines}, {interp_mode}, {dtype})")
pg.geocode(lt_fine, dem_fp, dem_width, ref_dem, rmli_width, out_lines, interp_mode, dtype)

create_png(ref_dem, rmli_par, image_dir)

#<================================== coregister secondary image(s) to reference =========================================>

del dates[ref_date]

for date, meta in dates.items():
    logger.info(f"Starting on {date}")

    ### multilook secondary image to amplitude image ###
    ## inputs ##
    # slc image and par
    sec_slc = proc_dir.joinpath(f"{date}.slc")
    sec_slc_par = proc_dir.joinpath(f"{date}.slc.par")
    ## outputs ##
    # reference multi looked intensity image
    smli = proc_dir.joinpath(date).with_suffix(".mli")
    smli_par = create_par_fp(smli)
    ## controls ##
    # 5 in range, 10 in azimuth
    multi_look = [5, 10]
    # multi look
    logger.info("pg.multi_look(ref_slc, sec_slc_par,smli, smli_par , *multi_look)")
    logger.info(f"pg.multi_look({sec_slc}, {sec_slc_par}, {smli}, {smli_par} , *{multi_look})")
    pg.multi_look(sec_slc, sec_slc_par, smli, smli_par , *multi_look)

    create_png(smli, smli_par, image_dir)

    ### Derive lookup table for SLC/MLI coregistration (considering terrain heights) ###
    ## inputs ##
    # mli1_par (reference)
    rmli_par
    # dem in reference image geoemtry (slant range)
    ref_dem
    # mli par for secondary image
    smli_par
    ## output (Lt tabel to resample mli2) to geometry of mli1 (fcomplex)
    lt_sec_ref = proc_dir.joinpath(f'{date}_{ref_date}.lt')

    logger.info(f"pg.rdc_trans(rmli_par, ref_dem, smli_par, lt_sec_ref)")
    logger.info(f"pg.rdc_trans({rmli_par}, {ref_dem}, {smli_par}, {lt_sec_ref})")
    pg.rdc_trans(rmli_par, ref_dem, smli_par, lt_sec_ref)

    ### resample secondary image to reference image
    # Resample SLC image using a lookup table and a refinement offset polynomial if available
    ## inputs ##
    # secondary slc image and slc image par
    sec_slc, sec_slc_par
    # reference slc image par
    ref_slc_par
    # lookup table from secondary to primary
    lt_sec_ref
    # reference multi looked parameter file
    rmli_par
    # secondary mli parameter file
    smli_par
    # offset parameter file ['-' for none]
    off_par = '-'
    ## outputs ##
    # slc2r - seondary slc resampled to reference geometry
    sec_slc_re = proc_dir.joinpath(f"{date}.slc.re")
    # slc2r par - parameter for resample slc
    sec_slc_re_par = proc_dir.joinpath(f"{date}.slc.re.par")


    logger.info("pg.SLC_interp_lt(sec_slc, ref_slc_par, sec_slc_par, lt_sec_ref, rmli_par, smli_par, off_par, sec_slc_re, sec_slc_re_par)")
    logger.info(f"pg.SLC_interp_lt({sec_slc}, {ref_slc_par}, {sec_slc_par}, {lt_sec_ref}, {rmli_par}, {smli_par}, {off_par}, {sec_slc_re}, {sec_slc_re_par})")
    pg.SLC_interp_lt(sec_slc, ref_slc_par, sec_slc_par, lt_sec_ref, rmli_par, smli_par, off_par, sec_slc_re, sec_slc_re_par)

    create_png(sec_slc_re, sec_slc_re_par, image_dir)

    # now we will use our secondary and reference images that are close and use amplitude cross correlation to refine lookup table

    ### Create and update ISP offset and interferogram parameter files ###
    ## inputs ##
    # slc1_par (reference ) <- for some reason the example uses the secondary for this argument?
    sec_slc_re_par
    # slc 2 parameter file
    sec_slc_re_par
    ## outputs ##
    # off_par (input/output) - output this time
    offset_par = proc_dir.joinpath(f'{date}.off_par')
    ## controls ##
    # algoirith 1 = intensity cross correlation
    matching_algorithm = 1
    # range looks [default = 1]
    rlooks = 1
    # azimuth looks [deafult = 1]
    azlooks = 1

    logger.info("pg.create_offset( sec_slc_re_par, sec_slc_re_par, offset_par, matching_algorithm, rlooks, azlooks, interactive)")
    logger.info(f"pg.create_offset( {sec_slc_re_par}, {sec_slc_re_par}, {offset_par}, {matching_algorithm}, {rlooks}, {azlooks}, {interactive})")
    pg.create_offset( sec_slc_re_par, sec_slc_re_par, offset_par, matching_algorithm, rlooks, azlooks, interactive)

    ### Offset estimation between MLI images using intensity cross-correlation ###
    # Offset estimation between MLI images using intensity cross-correlation of patches
    # compares reference MLI to secondary MLI amplitude image to improve coregistration
    ## inputs
    # intensity image 1
    rmli
    # intensity image 2
    smli
    # rmli par
    rmli_par
    # smli par
    smli_par
    # ISP offset/interferogram par file
    offset_par
    ## outputs ##
    # offset estimates in range and azimuth (fcomplex)
    sec_ref_offs = proc_dir.joinpath(f'{date}_{ref_date}.offs')
    # ccp 0 cross corrrelation of each path (0 -> 1)
    ccp = sec_ref_offs = proc_dir.joinpath(f'{date}_{ref_date}.ccp')
    ## controls ##
    # range patch size, azimuth patch size
    patch_n = (256, 512)
    # offsets and cc data in text format ['-' for no output]
    off_txt = '-'
    # slc oversampling factor [default = 2]
    n_ovr = 2
    # number of offset estimates in (range, azimuth)
    offset_samples = (32, 32)
    # cross correlatin threshold
    cc_thres = 0.1

    logger.info("pg.offset_pwr(ref_slc, sec_slc_re, ref_slc_par, sec_slc_re_par, offset_par, sec_ref_offs, ccp, *patch_n, off_txt, n_ovr, *offset_samples,cc_thres)")
    logger.info(f"pg.offset_pwr({ref_slc}, {sec_slc_re}, {ref_slc_par}, {sec_slc_re_par}, {offset_par}, {sec_ref_offs}, {ccp}, *{patch_n}, {off_txt}, {n_ovr}, *{offset_samples}, {cc_thres})")
    pg.offset_pwr(ref_slc, sec_slc_re, ref_slc_par, sec_slc_re_par, offset_par, sec_ref_offs, ccp,\
         *patch_n, off_txt, n_ovr, *offset_samples, cc_thres)

    ### Range and azimuth offset polynomial estimation using SVD from amplitude offsets ###
    # inputs #
    # offsets in azimuth and range
    sec_ref_offs
    # ccp - cross correlatin of each patch
    ccp
    # diff parameter file
    offset_par
    ## outputs ##
    #  culled range and azimuht offset estimates
    cull_offs = proc_dir.joinpath(f'{date}_{ref_date}').with_suffix('.coffs')
    # culled offset estimates and cross correlation values ['-' for no output]
    cull_offsets = '-'
    ## controls
    # cross correlation threshold
    cc_thres = 0.1
    # npoly - polynomila parameters default =4
    npoly = 4

    logger.info("pg.offset_fit(sec_ref_offs, ccp, offset_par, cull_offs, cull_offsets, cc_thres, npoly)")
    logger.info(f"pg.offset_fit({sec_ref_offs}, {ccp}, {offset_par}, {cull_offs}, {cull_offsets}, {cc_thres}, {npoly})")
    pg.offset_fit(sec_ref_offs, ccp, offset_par, cull_offs, cull_offsets, cc_thres, npoly)

    ### resample secondary image to reference image
    # Resample SLC image using a lookup table and now we can use the refinement offset from the amplitude cross correlation
    ## inputs ##
    # secondary slc image and slc image par
    sec_slc_re, sec_slc_re_par
    # reference slc image par
    ref_slc_par
    # lookup table from secondary to primary
    lt_sec_ref
    # reference multi looked parameter file
    rmli_par
    # secondary mli parameter file
    smli_par
    # offset parameter file ['-' for none]
    offset_par
    ## outputs ##
    # slc2r - seondary slc resampled to reference geometry
    sec_slc = proc_dir.joinpath(f"{date}.slc")
    # slc2r par - parameter for resample slc
    sec_slc_par = proc_dir.joinpath(f"{date}.slc.par")


    logger.info("pg.SLC_interp_lt(sec_slc_re, ref_slc_par, sec_slc_re_par, lt_sec_ref, rmli_par, smli_par, offset_par, sec_slc, sec_slc_par)")
    logger.info(f"pg.SLC_interp_lt({sec_slc_re}, {ref_slc_par}, {sec_slc_re_par}, {lt_sec_ref}, {rmli_par}, {smli_par}, {offset_par}, {sec_slc}, {sec_slc_par})")
    pg.SLC_interp_lt(sec_slc_re, ref_slc_par, sec_slc_re_par, lt_sec_ref, rmli_par, smli_par, offset_par, sec_slc, sec_slc_par)

    create_png(sec_slc, sec_slc_par, image_dir)

    ## multilook SLC again now that is coregistered to reference image ##
    ## inputs ##
    # slc image and par
    sec_slc
    sec_slc_par
    ## outputs ##
    # reference multi looked intensity image
    smli
    smli_par
    ## controls ##
    # 5 in range, 10 in azimuth
    multi_look = [5, 10]
    # multi look
    logger.info("pg.multi_look(ref_slc, sec_slc_par,smli, smli_par , *multi_look)")
    logger.info(f"pg.multi_look({sec_slc}, {sec_slc_par}, {smli}, {smli_par} , *{multi_look})")
    pg.multi_look(sec_slc, sec_slc_par, smli, smli_par , *multi_look)

    create_png(smli, smli_par, image_dir)

#<================================== generate differential interferogram =========================================>

    ### for the interferogram calculation an off_par file is needed ###
    ## inputs ##
    # slc1_par (reference ) <- for some reason the example uses the secondary for this argument?
    ref_slc_par
    # slc 2 parameter file
    sec_slc_par
    ## outputs ##
    # off_par (input/output) - output this time
    diff_par = out_dir.joinpath(f'{ref_date}_{date}.diff_par')
    ## controls ##
    # algoirith 1 = intensity cross correlation
    matching_algorithm = 1
    # range looks [default = 1]
    rlooks = 5
    # azimuth looks [deafult = 1]
    azlooks = 10

    logger.info("pg.create_offset( ref_slc_par, sec_slc_par, diff_par, matching_algorithm, rlooks, azlooks, interactive)")
    logger.info(f"pg.create_offset( {ref_slc_par}, {sec_slc_par}, {diff_par}, {matching_algorithm}, {rlooks}, {azlooks}, {interactive})")
    pg.create_offset( ref_slc_par, sec_slc_par, diff_par, matching_algorithm, rlooks, azlooks, interactive)    

    ### Use dem and orbit path to create simulated SAR image from geometry """"
    ## inputs ##
    # reference slc par
    ref_slc_par
    # secondary slc par (resampled to SLC1)
    sec_slc_par
    # differential dem parameter file
    diff_par
    # reference dem geocode to match reference image
    ref_dem
    # paraemter fiel of image used for geometric coregistration
    ref_slc_par
    # los deformation map ('-' for None), 'meters/yr', 'float'
    los = '-'
    # interferogram time interval (days)
    delta_t = '-'
    ## outputs ##
    # simulated phase
    sim_unw = proc_dir.joinpath(f'{ref_date}_{date}.sim_unw')
    ## controls ##
    # interferometric acquisition mode (enter - for default)
    # or 1 for repeat pass, 0 for tandem-x 
    int_acq_mode = 1
    # ph_mode - pahse offset mode 0 = aboslute phase [default], 1= substract phase offset that is multiple of 2pi
    ph_mode = 0
    
    logger.info("pg.phase_sim_orb(ref_slc_par, sec_slc_par, diff_par, ref_dem, sim_unw, ref_slc_par, los, delta_t, int_acq_mode, ph_mode)")
    logger.info(f"pg.phase_sim_orb({ref_slc_par}, {sec_slc_par}, {diff_par}, {ref_dem}, {sim_unw}, {ref_slc_par}, {los}, {delta_t}, {int_acq_mode}, {ph_mode})")
    pg.phase_sim_orb(ref_slc_par, sec_slc_par, diff_par, ref_dem, sim_unw, ref_slc_par, los, delta_t, int_acq_mode, ph_mode)

    create_png(sim_unw, diff_par, image_dir)

    ### run differential interferogram (multiply t1 by complex conjugate of t2) ###
    ## inputs ##
    # ref slc data
    ref_slc
    # secondary slc data (resampled to ref slc)
    sec_slc
    # ref par file
    ref_slc_par
    # secondary par file
    sec_slc_par
    # differential parameter file
    diff_par
    # simulated unwrapped phase from dem
    sim_unw
    ## outputs ##
    # differential phase between the two slcs
    diff = out_dir.joinpath(f'{ref_date}_{date}.int')
    ## controls ##
    # looks in range and azimuth
    looks = (5, 10)
    # sps_flag (range spectral shift flag)
    # 1 = apply range spectral shift [default]
    r_ss_flag = 1
    # azf_flag - azimuth common band filter flag
    #  1= apply common band filter flag [default]
    azf_flag = 1
    # rbw_min (min range bandwidth factor 0.1->1) [default = 0.25]
    rbw_min = 0.25 #0.1 in tutorial

    logger.info("pg.SLC_diff_intf(ref_slc, sec_slc, ref_slc_par, sec_slc_par, diff_par, sim_unw, diff, *looks, r_ss_flag, azf_flag, rbw_min)")
    logger.info(f"pg.SLC_diff_intf({ref_slc}, {sec_slc}, {ref_slc_par}, {sec_slc_par}, {diff_par}, {sim_unw}, {diff}, *{looks}, {r_ss_flag}, {azf_flag}, {rbw_min})")
    pg.SLC_diff_intf(ref_slc, sec_slc, ref_slc_par, sec_slc_par, diff_par, sim_unw, diff, *looks, r_ss_flag, azf_flag, rbw_min)

    create_png(diff, diff_par, image_dir)

    # calculate rasterfile showing differential interferogram
    diff_width = get_width(diff_par)
    pg.rasmph_pwr(diff, rmli, diff_width, 1, 0, 1, 1, 'rmg.cm', image_dir.joinpath(f'{ref_date}_{date}.diff.bmp'), 1., .35, 24)


    ### calculate coherence of differential phase ###
    ## inputs ##
    # complex interferogram
    diff
    # intensity image of reference scene
    rmli
    # intensity image of secondary scene
    smli
    # slope of phase
    phase_slope = '-'
    # texture (backscatter texture data)?
    texture = '-'
    ## outputs ##
    # coherence image (float)
    cc = out_dir.joinpath(f'{ref_date}_{date}.cor')
    # width 
    diff_width = get_width(diff_par)
    # smallest correlation average box size [default = 3]
    box_min = 3
    # largest correlation average box size [default = 9]
    box_max = 9

    logger.info("pg.cc_ad(diff, rmli, smli ,phase_slope, texture, cc, diff_width, box_min, box_max)")
    logger.info(f"pg.cc_ad({diff}, {rmli}, {smli} ,{phase_slope}, {texture}, {cc}, {diff_width}, {box_min}, {box_max})")
    pg.cc_ad(diff, rmli, smli ,phase_slope, texture, cc, diff_width, box_min, box_max)

    create_png(cc, diff_par, image_dir)

    # calculate raster file show coherence
    pg.ras_linear( cc, diff_width, 1, 0, 1, 1, 0.0, 1.0, 0, 'gray.cm', image_dir.joinpath(f'{ref_date}_{date}.cor.bmp'))


    ### georeferencing of the results ###
    ### secondary mli image ###
    ## inputs ##
    # input_data
    smli
    # input data width
    smli_width = get_width(smli_par)
    # lookup table
    lt_fine
    # out lines ['-' for number of lines in gc_map]
    out_lines = '-'
    # interp mode  3 = bicubic-sqrt spline
    interp_mode = 7
    # dtype # 0 for float
    dtype = 0 
    ## outputs ##
    # georeferences secondary mli
    smli_geo = proc_dir.joinpath(date).with_suffix(".smli.geo")

    pg.geocode_back(smli, smli_width, lt_fine, smli_geo, dem_width, out_lines, interp_mode, dtype)
    create_png(smli_geo, dem_par, image_dir)

    ### georeferencing of coherence ##
    ## outputs ##
    # georeferences coherence
    cc_geo = out_dir.joinpath(f'{ref_date}_{date}.cor.geo')

    pg.geocode_back( cc, diff_width, lt_fine,  cc_geo, dem_width, out_lines, interp_mode, dtype)
    create_png(cc_geo, dem_par, image_dir)
    # calculate raster file show coherence
    cor_bmp = image_dir.joinpath(f'{ref_date}_{date}.cor.geo.bmp')
    pg.ras_linear( cc_geo, dem_width, 1, 0, 1, 1, 0.0, 1.0, 0, 'gray.cm', cor_bmp)
    cor_kml = image_dir.joinpath(f'{ref_date}_{date}.cor.geo.kml')
    pg.kml_map(cor_bmp, dem_par, cor_kml)


    ### georeferencing of phase ###
    ## inputs ##
    # dtype = fcomplex
    dtype = 1
    # interp mode = Lannczos interploation
    interp_mode = 6
    ## outputs ##
    diff_geo = out_dir.joinpath(f'{ref_date}_{date}.int.geo')
    pg.geocode_back(diff, diff_width,lt_fine, diff_geo, dem_width, out_lines, interp_mode, dtype)

    create_png(diff_geo, dem_par, image_dir)
    
    diff_bmp = image_dir.joinpath(f'{ref_date}_{date}.diff.geo.bmp')
    pg.rasmph_pwr(diff_geo, rmli_geo, dem_width, 1, 0, 1, 1, 'rmg.cm', diff_bmp, 1., .35, 24)
    diff_kml = image_dir.joinpath(f'{ref_date}_{date}.diff.geo.kml')
    pg.kml_map(diff_bmp, dem_par, diff_kml)

