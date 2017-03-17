#!/bin/python

import datetime as dt

import numpy as np
from osgeo import gdal, gdal_array
import pandas as pd
import yaml
import click

import yatsm
from yatsm.io import read_line
from yatsm.utils import csvfile_to_dataframe, get_image_IDs
import yatsm._cyprep as cyprep

@click.command()
@click.argument('row1', metavar='<row1>', nargs=1, type=click.INT)
@click.argument('row2', metavar='<row2>', nargs=1, type=click.INT)


def annual(row1, row2):
	NDV = -9999   

	# EXAMPLE IMAGE for dimensions, map creation
	example_img_fn = '/projectnb/landsat/users/valpasq/LCMS/stacks/p045r030/images/example_img'
	# YATSM CONFIG FILE
	config_file = '/projectnb/landsat/users/valpasq/LCMS/stacks/p045r030/p045r030_config_LCMS.yaml'

	WRS2 = 'p045r030'

	# Up front -- declare hard coded dataset attributes (for now)
	BAND_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'therm', 'tcb', 'tcg', 'tcw', 'fmask']
	n_band = len(BAND_NAMES) - 1
	col_names = ['date', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'therm', 'tcb', 'tcg', 'tcw']
	dtype = np.int16
	years = range(1985, 2016, 1)
	length = 33 # number of years 

	# Set percentiles for range calculations
	# TODO - make this a click input
	pct1=0.05
	pct2=0.95

	# Read in example image
	example_img = read_image(example_img_fn)
	py_dim = example_img.shape[0]
	px_dim = example_img.shape[1]
	print('Shape of example image:')
	print(example_img.shape)

	# Read in and parse config file
	cfg = yaml.load(open(config_file))
	# List to np.ndarray so it works with cyprep.get_valid_mask
	cfg['dataset']['min_values'] = np.asarray(cfg['dataset']['min_values'])
	cfg['dataset']['max_values'] = np.asarray(cfg['dataset']['max_values'])

	# Get files list
	df = csvfile_to_dataframe(cfg['dataset']['input_file'], \
	                          date_format=cfg['dataset']['date_format'])

	# Get dates for image stack
	df['image_ID'] = get_image_IDs(df['filename']) 
	df['x'] = df['date'] 
	dates = df['date'].values

	# Initialize arrays for storing stats and disturbance events
	mean_TCB = np.zeros((py_dim, px_dim, length))
	mean_TCG = np.zeros((py_dim, px_dim, length))
	mean_TCW = np.zeros((py_dim, px_dim, length))

	min_TCB = np.zeros((py_dim, px_dim, length))
	min_TCG = np.zeros((py_dim, px_dim, length))
	min_TCW = np.zeros((py_dim, px_dim, length))

	max_TCB = np.zeros((py_dim, px_dim, length))
	max_TCG = np.zeros((py_dim, px_dim, length))
	max_TCW = np.zeros((py_dim, px_dim, length))

	range_TCB = np.zeros((py_dim, px_dim, length))
	range_TCG = np.zeros((py_dim, px_dim, length))
	range_TCW = np.zeros((py_dim, px_dim, length))

	for py in range(row1, row2): # row iterator
	    print('Working on row {py}'.format(py=py))
	    Y_row = read_line(py, df['filename'], df['image_ID'], cfg['dataset'],
	                      px_dim, n_band + 1, dtype,  # +1 for now for Fmask
	                      read_cache=False, write_cache=False,
	                      validate_cache=False)
	    
	    for px in range(0, px_dim): # column iterator
	        Y = Y_row.take(px, axis=2)
	        
	        if (Y[0:6] == NDV).mean() > 0.3:
	            continue
	        else: # process time series for disturbance events
	            
				# Mask based on physical constraints and Fmask 
				valid = cyprep.get_valid_mask( \
				            Y, \
				            cfg['dataset']['min_values'], \
				            cfg['dataset']['max_values']).astype(bool)

				# Apply mask band
				idx_mask = cfg['dataset']['mask_band'] - 1
				valid *= np.in1d(Y.take(idx_mask, axis=0), \
				                         cfg['dataset']['mask_values'], \
				                         invert=True).astype(np.bool)

				# Mask time series using fmask result
				Y_fmask = np.delete(Y, idx_mask, axis=0)[:, valid]
				dates_fmask = dates[valid]

				# Apply multi-temporal mask (modified tmask)
				# Step 1. mask where green > 3 stddev from mean (fmasked) green
				multitemp1_fmask = np.where(Y_fmask[1] < (np.mean(Y_fmask[1])+np.std(Y_fmask[1])*3))
				dates_fmask = dates_fmask[multitemp1_fmask[0]] 
				Y_fmask = Y_fmask[:, multitemp1_fmask[0]]
				# Step 2. mask where swir < 3 std devfrom mean (fmasked) SWIR
				multitemp2_fmask = np.where(Y_fmask[4] > (np.mean(Y_fmask[4])-np.std(Y_fmask[4])*3))
				dates_fmask = dates_fmask[multitemp2_fmask[0]] 
				Y_fmask = Y_fmask[:, multitemp2_fmask[0]]

				# convert time from ordinal to dates
				dt_dates_fmask = np.array([dt.datetime.fromordinal(d) for d in dates_fmask])

				# Create dataframes for analysis 
				# Step 1. reshape data
				shp_ = dt_dates_fmask.shape[0]
				dt_dates_fmask_csv = dt_dates_fmask.reshape(shp_, 1)
				Y_fmask_csv = np.transpose(Y_fmask)
				data_fmask = np.concatenate([dt_dates_fmask_csv, Y_fmask_csv], axis=1)
				# Step 2. create dataframe
				data_fmask_df = pd.DataFrame(data_fmask, columns=col_names)
				# convert reflectance to numeric type
				data_fmask_df[BAND_NAMES[0:10]] = data_fmask_df[BAND_NAMES[0:10]].astype(int) 

				# Group observations by year to generate annual TS
				year_group_fmask = data_fmask_df.groupby(data_fmask_df.date.dt.year)
				# get years in time series 
				years_fmask = np.asarray(year_group_fmask.groups.keys()) 
				years_fmask = years_fmask.astype(int)

				# Calculate number of observations
				nobs = year_group_fmask['tcb'].count()

				### TC Brightness
				# Calculate mean annual TCB
				TCB_mean = year_group_fmask['tcb'].mean()
				# percentile clip 
				TCB_iqr_H = year_group_fmask['tcb'].quantile([pct2])
				TCB_iqr_L = year_group_fmask['tcb'].quantile([pct1])

				### TC Greenness 
				# Calculate mean annual TCG
				TCG_mean = year_group_fmask['tcg'].mean() 
				# percentile clip 
				TCG_iqr_H = year_group_fmask['tcg'].quantile([pct2])
				TCG_iqr_L = year_group_fmask['tcg'].quantile([pct1])

				### TC Wetness 
				# Calculate mean annual TCW
				TCW_mean = year_group_fmask['tcw'].mean()
				# percentile clip 
				TCW_iqr_H = year_group_fmask['tcw'].quantile([pct2])
				TCW_iqr_L = year_group_fmask['tcw'].quantile([pct1])


				for index, year in enumerate(years):             
				    if year in TCB_mean.index: 
				        mean_TCB[py, px, index] = TCB_mean[year]
				        mean_TCG[py, px, index] = TCG_mean[year]
				        mean_TCW[py, px, index] = TCW_mean[year]

				        min_TCB[py, px, index] = TCB_iqr_L[year]
				        min_TCG[py, px, index] = TCG_iqr_L[year]
				        min_TCW[py, px, index] = TCW_iqr_L[year]

				        max_TCB[py, px, index] = TCB_iqr_H[year]
				        max_TCG[py, px, index] = TCG_iqr_H[year]
				        max_TCW[py, px, index] = TCW_iqr_H[year]

	print('Statistics complete')

	# Output map for each year
	in_ds = gdal.Open(example_img_fn, gdal.GA_ReadOnly)

	for index, year in enumerate(years): 
	    condition_fn = './{WRS2}_ST-BGW_mean_{year}_{row1}-{row2}.tif'.format(WRS2=WRS2, year=year, row1=row1, row2=row2)
	    out_driver = gdal.GetDriverByName("GTiff")
	    out_ds = out_driver.Create(condition_fn, 
	                               example_img.shape[1],  # x size
	                               example_img.shape[0],  # y size
	                               3,  # number of bands
	                               gdal.GDT_Int32)
	    out_ds.SetProjection(in_ds.GetProjection())
	    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
	    out_ds.GetRasterBand(1).WriteArray(mean_TCB[:, :, index])
	    out_ds.GetRasterBand(1).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(1).SetDescription('Mean Annual TC Brightness')
	    out_ds.GetRasterBand(2).WriteArray(mean_TCG[:, :, index])
	    out_ds.GetRasterBand(2).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(2).SetDescription('Mean Annual TC Greenness')
	    out_ds.GetRasterBand(3).WriteArray(mean_TCW[:, :, index])
	    out_ds.GetRasterBand(3).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(3).SetDescription('Mean Annual TC Wetness')
	    out_ds = None

	    condition_fn = './{WRS2}_ST-BGW_min_{year}_{row1}-{row2}.tif'.format(WRS2=WRS2, year=year, row1=row1, row2=row2)
	    out_driver = gdal.GetDriverByName("GTiff")
	    out_ds = out_driver.Create(condition_fn, 
	                               example_img.shape[1],  # x size
	                               example_img.shape[0],  # y size
	                               3,  # number of bands
	                               gdal.GDT_Int32)
	    out_ds.SetProjection(in_ds.GetProjection())
	    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
	    out_ds.GetRasterBand(1).WriteArray(min_TCB[:, :, index])
	    out_ds.GetRasterBand(1).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(1).SetDescription('Minimum Annual TC Brightness')
	    out_ds.GetRasterBand(2).WriteArray(min_TCG[:, :, index])
	    out_ds.GetRasterBand(2).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(2).SetDescription('Minimum Annual TC Greenness')
	    out_ds.GetRasterBand(3).WriteArray(min_TCW[:, :, index])
	    out_ds.GetRasterBand(3).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(3).SetDescription('Minimum Annual TC Wetness')
	    out_ds = None

	    condition_fn = './{WRS2}_ST-BGW_max_{year}_{row1}-{row2}.tif'.format(WRS2=WRS2, year=year, row1=row1, row2=row2)
	    out_driver = gdal.GetDriverByName("GTiff")
	    out_ds = out_driver.Create(condition_fn, 
	                               example_img.shape[1],  # x size
	                               example_img.shape[0],  # y size
	                               3,  # number of bands
	                               gdal.GDT_Int32)
	    out_ds.SetProjection(in_ds.GetProjection())
	    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
	    out_ds.GetRasterBand(1).WriteArray(max_TCB[:, :, index])
	    out_ds.GetRasterBand(1).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(1).SetDescription('Maximum Annual TC Brightness')
	    out_ds.GetRasterBand(2).WriteArray(max_TCG[:, :, index])
	    out_ds.GetRasterBand(2).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(2).SetDescription('Maximum Annual TC Greenness')
	    out_ds.GetRasterBand(3).WriteArray(max_TCW[:, :, index])
	    out_ds.GetRasterBand(3).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(3).SetDescription('Maximum Annual TC Wetness')
	    out_ds = None

	    condition_fn = './{WRS2}_ST-BGW_range_{year}_{row1}-{row2}.tif'.format(WRS2=WRS2, year=year, row1=row1, row2=row2)
	    out_driver = gdal.GetDriverByName("GTiff")
	    out_ds = out_driver.Create(condition_fn, 
	                               example_img.shape[1],  # x size
	                               example_img.shape[0],  # y size
	                               3,  # number of bands
	                               gdal.GDT_Int32)
	    out_ds.SetProjection(in_ds.GetProjection())
	    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
	    out_ds.GetRasterBand(1).WriteArray(range_TCB[:, :, index])
	    out_ds.GetRasterBand(1).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(1).SetDescription('Annual Range TC Brightness')
	    out_ds.GetRasterBand(2).WriteArray(range_TCG[:, :, index])
	    out_ds.GetRasterBand(2).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(2).SetDescription('Annual Range TC Greenness')
	    out_ds.GetRasterBand(3).WriteArray(range_TCW[:, :, index])
	    out_ds.GetRasterBand(3).SetNoDataValue(-9999)
	    out_ds.GetRasterBand(3).SetDescription('Annual Range TC Wetness')
	    out_ds = None

# Define image reading function
def read_image(f):
    ds = gdal.Open(f, gdal.GA_ReadOnly)
    return ds.GetRasterBand(1).ReadAsArray()

if __name__ == '__main__':
    annual()