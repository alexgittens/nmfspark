export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cscratch1/sd/gittens/nmf-spark/lib

sbatch runnmf_50.slurm 
sbatch runnmf_101.slurm
sbatch runnmg_301.slurm
