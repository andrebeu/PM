#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a stsize_arr=(25 40 55)
declare -a num_pmtrials_arr=(2 4 6)
declare -a pm_weight_arr=(2 4 6)

for seed in {10..15}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for num_pmtrials in "${num_pmtrials_arr[@]}"; do 
			for pm_weight in "${pm_weight_arr[@]}"; do 
				sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${stsize}" "${num_pmtrials}" "${pm_weight}"
			done
		done
	done
done
