#!/bin/bash
#$ -V
#$ -l h_rt=24:00:00
#$ -N ST_merge
#$ -j y

echo "merging ST results"

WRS2=$1
feat=$2

for year in $(seq 1985 2015); do
	echo $year	
	files=$(find $d -name "*$WRS2*$feat*$year*" -type f | sort)
	qsub -j y -V -l h_rt=24:00:00 -N ${feat}_${year} -b y \
		gdal_merge.py -n 0 -a_nodata -9999 -o ./${WRS2}_ST-BGW_${feat}_${year}.tif $files
done
