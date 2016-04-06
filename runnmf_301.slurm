#!/bin/bash -l

#SBATCH -N 301
#SBATCH -p regular
#SBATCH -t 5
#SBATCH --ccm
#SBATCH -A m1523
#SBATCH --reservation=INC0082872

module load spark/1.5.1 sbt collectl
export SPARK_LOCAL_DIRS="/tmp,/dev/shm"
start-collectl.sh
start-all.sh
sbt nodes301
stop-all.sh
stop-collectl.sh
