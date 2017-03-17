#!/bin/bash
#$ -V
#$ -l h_rt=24:00:00
#$ -N annualstats
#$ -j y

qsub annual.sh 0 500
qsub annual.sh 500 1000
qsub annual.sh 1000 1500
qsub annual.sh 1500 2000
qsub annual.sh 2000 2500
qsub annual.sh 2500 3000
qsub annual.sh 3000 3500
qsub annual.sh 3500 4000
qsub annual.sh 4000 4500
qsub annual.sh 4500 5000
qsub annual.sh 5000 5500
qsub annual.sh 5500 6000
qsub annual.sh 6000 6500
qsub annual.sh 6500 7000
qsub annual.sh 7000 7170
