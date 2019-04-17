
declare -a stsize_arr=(10)


for seed in {100..115}; do 
	for stsize in "${stsize_arr[@]}"; do 
		echo ${seed} ${stsize}
		python pmtask.py ${seed} "purewm" ${stsize}
	done
done