#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name=WR_HMC_Real
#SBATCH --time=72:00:00
#SBATCH -o HPC/run_2/output
#SBATCH -e HPC/run_2/errors

module load anaconda3

source activate
conda activate WRModelHMC

python Apep_HMC_HPC.py