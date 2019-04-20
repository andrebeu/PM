#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a signal_arr=(7 8)
declare -a pmweight_arr=(1 2 3 4)

for seed in {5..25}; do 
	for signal in "${signal_arr[@]}"; do 
		for pmweight in "${pmweight_arr[@]}"; do 
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${signal}" "${pmweight}" "0" "5"
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${signal}" "${pmweight}" "1" "5"
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${signal}" "${pmweight}" "0" "0"
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${signal}" "${pmweight}" "1" "0"
		done
	done
done
