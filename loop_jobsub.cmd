#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a stsize_arr=(20)
declare -a pmtrials_arr=(2 3 4 6)

for seed in {10..20}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for pmtrials in "${pmtrials_arr[@]}"; do 
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "purewm" "${stsize}" "${pmtrials}" "1"
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "purewm" "${stsize}" "${pmtrials}" "2"
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "purewm" "${stsize}" "${pmtrials}" "3"
		done
	done
done
