#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/pm"
##

declare -a nmaps_arr=(6 8)
declare -a ntrials_arr=(5 10)
declare -a switch_arr=(1 0)
declare -a wmsize_arr=(4 6)
declare -a trlen_arr=(6 10)

for seed in {0..19}; do 
  for nmaps in "${nmaps_arr[@]}"; do 
    for ntrials in "${ntrials_arr[@]}"; do 
      for switch in "${switch_arr[@]}"; do 
        for wmsize in "${wmsize_arr[@]}"; do 
          for trlen in "${trlen_arr[@]}"; do 
            sbatch ${wd_dir}/gpu_jobsub.cmd ${wmsize} ${nmaps} ${switch} ${ntrials} ${trlen} ${seed}
          done
        done
      done
    done
	done
done
