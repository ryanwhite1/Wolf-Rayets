#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name=WR_HMC_Simulated
#SBATCH --time=72:00:00
#SBATCH --nodelist=smp-7-4
#SBATCH -o HPC/cavity_sim_run_1/output
#SBATCH -e HPC/cavity_sim_run_1/errors

module load anaconda3

source activate
conda activate WRModelHMC

python Apep_HMC_Simulated_HPC_Cavity.py