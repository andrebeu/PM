#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


# declare -a ntrials_arr=(1 2 3)
declare -a em_arr=(0 1)
declare -a nmaps_arr=(4 5 6 8)

for seed in {0..29}; do 
  for em in "${em_arr[@]}"; do 
    for nmaps in "${nmaps_arr[@]}"; do 
 #      for stsize2 in "${stsize2_arr[@]}"; do 
      sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${em} ${nmaps} 
 #      done
    done
	done
done
