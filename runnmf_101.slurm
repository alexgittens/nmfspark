#!/bin/bash -l

#SBATCH -N 101
#SBATCH -p regular
#SBATCH -t 5
#SBATCH --ccm

module load spark/1.5.1 sbt
export SPARK_LOCAL_DIRS="/tmp,/dev/shm"
start-all.sh
sbt nodes101
