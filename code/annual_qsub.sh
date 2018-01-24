#!/bin/bash
#$ -V
#$ -l h_rt=24:00:00
#$ -N annualstats
#$ -j y

start=$1
step=25
stop=$2
end=$3

# run annual stats for image chunks
for row1 in $(seq $start $step $stop); do
	row2=`expr $row1 + $step`
	qsub /projectnb/landsat/users/valpasq/LCMS/dataviz/code/annual.sh $row1 $row2
done

#run last odd rows
qsub /projectnb/landsat/users/valpasq/LCMS/dataviz/code/annual.sh $stop $end
