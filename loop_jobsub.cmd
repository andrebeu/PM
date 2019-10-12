#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a nmaps_arr=(4 5 6)
declare -a ntrials_arr=(2 3 4 6)


for seed in {0..19}; do 
  for nmaps in "${nmaps_arr[@]}"; do 
    for ntrials in "${ntrials_arr[@]}"; do 
      sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${nmaps} ${ntrials} 
    done
	done
done
