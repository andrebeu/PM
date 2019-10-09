#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


declare -a instdim_arr=(6 8 10 12)
declare -a stimdim_arr=(6 8 10 12)
declare -a stsize_arr=(8 10 12 14)
declare -a deep_arr=(0 1)
declare -a wmsetting_arr=(0 1)

for seed in {0..19}; do 
  for instdim in "${instdim_arr[@]}"; do 
    for stimdim in "${stimdim_arr[@]}"; do 
      for stsize in "${stsize_arr[@]}"; do 
        for deep in "${deep_arr[@]}"; do 
          for wmsetting in "${wmsetting_arr[@]}"; do 
            sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${instdim} ${stimdim} ${stsize} ${deep} ${wmsetting}
          done
        done
      done
    done
	done
done
