#!/bin/bash

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=abeukers@princeton.edu

#SBATCH -t 40:00:00			# runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1				# node count 
#SBATCH -c 16				# number of cores 

seed=${1}
emsetting=${2}
ntrials=${3}
trlen=${4}

module load anaconda3/4.4.0
module load cudnn/cuda-8.0/6.0

printf "\n\n complex maps task \n\n"

srun python -u "/tigress/abeukers/wd/pm/exp-amtask-sweep1.py" ${seed} ${emsetting} ${ntrials} ${trlen} 


sacct --format="CPUTime,MaxRSS"