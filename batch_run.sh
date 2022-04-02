#!/bin/bash

# distribute
lamda=(0.01 0.001)
gpu=(3 4)

# iterate
sample_weight_mul=(1 1.2 1.4)
sample_weight_add=(0 0.001 0.01 0.1 0.3)

for sample_weight_mul in "${sample_weight_mul[@]}"
do
	for sample_weight_add in "${sample_weight_add[@]}"
		do
		python main.py --epochs 500 --polydecay --gpu ${gpu[0]} --loss bce-r --lr_reduction_factor 10 --lr_decay_steps 80000 --is_green 1 --final_activation sigmoid --lamda ${lamda[0]} --random_seed 0 --sample_weight_mul sample_weight_mul -sample_weight_add sample_weight_add &
		python main.py --epochs 500 --polydecay --gpu ${gpu[1]} --loss bce-r --lr_reduction_factor 10 --lr_decay_steps 80000 --is_green 1 --final_activation sigmoid --lamda ${lamda[1]} --random_seed 0 --sample_weight_mul sample_weight_mul -sample_weight_add sample_weight_add &
		wait
		
		done
done
