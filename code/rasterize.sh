#!/bin/bash

if [ -z "$1" ]; then
    echo "Error - please specify referemce image. Usage:"
    echo "    $0 <path of image>"
    exit 1
fi

img=$1 # example image
shp='./FIA_data/forestedFIA_public_albers.shp' # shapefile
lyr='forestedFIA_public_albers' # layer name (typically shapefile name without .shp extension)
output='./FIA_FORTYPCD.tif' # name for raster output

# Get extent from example image
EXTENT=$(gdalinfo $img |\
        grep "Upper Left\|Lower Right" |\
        sed "s/Upper Left  //g;s/Lower Right //g;s/).*//g" |\
        tr "\n" " " |\
        sed 's/ *$//g' |\
        tr -d "[(,]")
echo -n "$EXTENT"

# Convert to warp format
warp_ext=$(echo $EXTENT | awk '{ print $1 " " $4 " " $3 " " $2 }')
echo "gdalwarp extent:"
echo $warp_ext

# Rasterize shapefile
gdal_rasterize -te $warp_ext -tr 30 -30 \
    -a FORTYPCD -l $lyr \
    -a_nodata -9999 -init -9999 -ot Int32 \
    -co "COMPRESS=LZW" -a_srs EPSG:5070 \
    $shp $output