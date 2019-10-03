#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


# declare -a ntrials_arr=(1 2 3)
# declare -a stsize1_arr=(10 30)
# declare -a stsize2_arr=(30 50)

for seed in {0..29}; do 
 #  for ntrials in "${ntrials_arr[@]}"; do 
 #    for stsize1 in "${stsize1_arr[@]}"; do 
 #      for stsize2 in "${stsize2_arr[@]}"; do 
  sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} 
 #      done
 #    done
	# done
done
