#!/bin/bash

#SBATCH --gpu-accounting

#SBATCH -t 48:00:00			# runs for 48 hours (max)  
#SBATCH -c 8				# number of cores 4
#SBATCH -N 1				# node count 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:0		# number of gpus 4

printf "\n\n\n --ntasks-per-node=1 -c=8 ntasks-per-socket=4 \n\n\n"


seed=${1}
stsize=${2} 
ntokens=${3} 
seqlen=${4} 


module load anaconda3/4.4.0
module load cudnn/cuda-9.1/7.1.2

printf "\n\n PI Network + PM Task \n\n"

srun python -u "/tigress/abeukers/wd/pm/pisims.py" ${seed} ${stsize} ${ntokens} ${seqlen} 


printf "\n\nGPU profiling \n\n"
sacct --format="elapsed,CPUTime,TotalCPU"
nvidia-smi --query-accounted-apps=gpu_serial,gpu_utilization,mem_utilization,max_memory_usage --format=csv
