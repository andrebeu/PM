#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##


# declare -a ntrials_arr=(1 2 3)
declare -a em_arr=(0 1)
declare -a ntrials_arr=(3)
declare -a trlen_arr=(20 30 40)

for seed in {0..19}; do 
  for em in "${em_arr[@]}"; do 
    for ntrials in "${ntrials_arr[@]}"; do 
      for trlen in "${trlen_arr[@]}"; do 
        sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${em} ${ntrials} ${trlen} 
      done
    done
	done
done
