#!/bin/bash
#$ -V
#$ -l h_rt=24:00:00
#$ -N annualstats
#$ -j y

start=0
step=100
stop=7100
end=7170

# run annual stats for image chunks
for row1 in $(seq $start $step $stop); do
	row2=`expr $row1 + $step`
	qsub annual.sh -j y -V -l h_rt=48:00:00 -N stats_$row1 -b y \
	$row1 $row2

done

# run last odd rows
qsub annual.sh $row2 $stop