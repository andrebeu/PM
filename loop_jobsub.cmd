#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a focal_arr=(0 1)
declare -a pmweight_arr=(1 2 3)

for seed in {10..20}; do 
	for focal in "${focal_arr[@]}"; do 
		for pmweight in "${pmweight_arr[@]}"; do 
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${focal}" "${pmweight}"
		done
	done
done
