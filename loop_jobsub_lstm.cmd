#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


declare -a switch_arr=(0 1)

for seed in {0..19}; do 
  for ntrials in "${switch_arr[@]}"; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${switch} 
	done
done
