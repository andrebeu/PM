#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


declare -a switch_arr=(1 0)

for seed in {0..99}; do 
  for switch in "${switch_arr[@]}"; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${switch}
	done
done
