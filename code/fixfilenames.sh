#!/bin/bash
#$ -V
#$ -l h_rt=24:00:00
#$ -N ST_merge
#$ -j y

for year in $(seq 1985 2015); do
	mv ./p045r032_ST-BGW_max_mon_${year}.tif ./p045r030_ST-BGW_max_mon_${year}.tif
	mv ./p045r032_ST-BGW_max_val_${year}.tif ./p045r030_ST-BGW_max_val_${year}.tif
	mv ./p045r032_ST-BGW_min_mon_${year}.tif ./p045r030_ST-BGW_min_mon_${year}.tif
	mv ./p045r032_ST-BGW_min_val_${year}.tif ./p045r030_ST-BGW_min_val_${year}.tif
done