#!/bin/bash

#SBATCH --job-name=gradient_boost_job   ## job name
#SBATCH -p free                         ## use free partition
#SBATCH --nodes=1                       ## use 1 node, don't ask for multiple
#SBATCH --ntasks=8                      ## ask for 8 CPU
#SBATCH --mem-per-cpu=4G                
#SBATCH --error=%x.%A.err               ## Slurm error  file, %x - job name, %A job id
#SBATCH --out=%x.%A.out                 ## Slurm output file, %x - job name, %A job id


# load python module
module purge
module load anaconda/2022.05
# module load python/3.8.0


# # Activate your personal Python environment
source /data/homezvol2/ddlin/mambaforge-pypy3/envs/cnvqc/lib/python3.8/venv/scripts/common/activate

# Run your Python script
python gradient_boost.py

