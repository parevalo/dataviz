#!/bin/bash
#$ -V
#$ -l h_rt=72:00:00
#$ -N annualstats
#$ -j y

row1=$1
row2=$2

python annual.py $row1 $row2
