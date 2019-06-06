#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a stsize_arr=(15 25 40)
declare -a ntokens_arr=(2 3 4 5)
declare -a seqlen_arr=(4 10 15)

for seed in {0..10}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for ntokens in "${ntokens_arr[@]}"; do 
			for seqlen in "${seqlen_arr[@]}"; do 
				sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${stsize} ${ntokens} ${seqlen} 
			done
		done
	done
done
