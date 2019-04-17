
declare -a stsize_arr=(2 5 15 20)


for seed in {10..15}; do 
	for stsize in "${stsize_arr[@]}"; do 
		echo ${seed} ${stsize}
		python pmtask.py ${seed} "purewm" ${stsize}
	done
done