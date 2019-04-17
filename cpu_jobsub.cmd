#!/bin/bash

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=abeukers@princeton.edu

#SBATCH -t 24:00:00			# runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1				# node count 
#SBATCH -c 16				# number of cores 

printf "tiger CPU "


seed=${1}
arch=${2}
stsize=${3}

module load anaconda3/4.4.0
module load cudnn/cuda-9.1/7.1.2
module load openmpi/gcc/2.1.0/64 # srm


printf "\n\n nback Task \n\n"

srun python -u "/tigress/abeukers/wd/pm/pmtask.py" ${seed} ${arch} ${stsize}

