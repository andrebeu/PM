#!/bin/bash

#SBATCH --gpu-accounting

#SBATCH -t 6:00:00			# runs for 48 hours (max)  
#SBATCH -c 8				# number of cores 4
#SBATCH -N 1				# node count 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1		# number of gpus 4

seed=${1}
instdim=${2}
stimdim=${3}
stsize=${4}
deep=${5}
wmsetting=${6}

module load anaconda3/4.4.0
module load cudnn/cuda-9.1/7.1.2

printf "\n\n MINI complex maps task \n\n"
 
srun python -u "/tigress/abeukers/wd/pm/exp-amtask-mini.py" ${seed} ${instdim} ${stimdim} ${stsize} ${deep} ${wmsetting}

printf "\n\nGPU profiling \n\n"
sacct --format="elapsed,CPUTime,TotalCPU"
nvidia-smi --query-accounted-apps=gpu_serial,gpu_utilization,mem_utilization,max_memory_usage --format=csv
