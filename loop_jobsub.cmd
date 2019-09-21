#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


declare -a switch_arr=(1 0)
declare -a nback_arr=(1 2 3)
declare -a nmaps_arr=(3 4 5 8)

for seed in {0..20}; do 
  for nback in "${nback_arr[@]}"; do 
    for nmaps in "${nmaps_arr[@]}"; do 
  		sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${nback} ${nmaps} 
    done
	done
done
