#!/bin/bash

#SBATCH -t 05:00:00			# runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1				# node count 
#SBATCH -c 28				# number of cores 

printf "tiger CPU "

seed=${1}
arch=${2}
stsize=${3}
pmtrials=${4}
lossweight=${5}

module load anaconda3/4.4.0

printf "\n\n PM Task \n\n"

srun python -u "/tigress/abeukers/wd/pm/pmtask.py" ${seed} ${arch} ${stsize} ${pmtrials} ${lossweight}

