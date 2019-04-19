#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a stsize_arr=(20 30 40)
declare -a num_pmtrials_arr=(4 5 6)
declare -a pm_weight_arr=(1)
declare -a nback_arr=(1 2)

for seed in {5..15}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for num_pmtrials in "${num_pmtrials_arr[@]}"; do 
			for pm_weight in "${pm_weight_arr[@]}"; do 
				for nback in "${nback_arr[@]}"; do 
					sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${stsize}" "${num_pmtrials}" "${pm_weight}" "0" "${nback}" 
					sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${stsize}" "${num_pmtrials}" "${pm_weight}" "1" "${nback}"
				done
			done
		done
	done
done
