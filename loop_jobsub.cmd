#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a stsize_arr=(6 8 12 14)

for seed in {10..15}; do 
	for stsize in "${stsize_arr[@]}"; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "purewm" "${stsize}"
	done
done
