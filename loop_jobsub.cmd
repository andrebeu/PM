#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


declare -a switch_arr=(1 0)
declare -a ntokens_arr=(3 4 5 6)
declare -a seqlen_arr=(3 4 5 6 10)
declare -a ntrials_arr=(2 3 4)

for seed in {0..20}; do 
	for switch in "${switch_arr[@]}"; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${switch} ${ntokens} ${seqlen} ${ntrials}
	done
done
