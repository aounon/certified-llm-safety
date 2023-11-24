#!/bin/bash

for max_erase in 0 4 8 12
do
    python main.py \
        --num_prompts 120 \
        --mode insertion \
        --max_erase $max_erase \
        --num_adv 1 \
        --use_classifier \
        --safe_prompts data/safe_prompts_test.txt \
        --model_wt_path models/insertion_distillbert_saved_weights.pt
done

# for num_adv in 1 2
# do
#     for max_erase in 0 2 4 6
#     do
#         python main.py --num_prompts 30 --mode insertion --max_erase $max_erase --num_adv $num_adv --use_classifier
#     done
# done