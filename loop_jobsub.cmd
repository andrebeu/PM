#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


declare -a switch_arr=(1 0)
declare -a ntokens_arr=(10)
declare -a seqlen_arr=(10 15)
declare -a ntrials_arr=(10)

for seed in {0..9}; do 
	for switch in "${switch_arr[@]}"; do 
    for ntokens in "${ntokens_arr[@]}"; do 
      for seqlen in "${seqlen_arr[@]}"; do 
        for ntrials in "${ntrials_arr[@]}"; do 
      		sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${switch} ${ntokens} ${seqlen} ${ntrials}
        done
      done
    done
	done
done
