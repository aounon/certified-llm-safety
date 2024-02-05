#!/bin/bash

# Llama 2
for max_erase in 0 4 8 12
do
    python main.py \
        --num_prompts 200 \
        --mode insertion \
        --eval_type safe \
        --max_erase $max_erase \
        --num_adv 1 \
        --safe_prompts data/safe_prompts.txt
done

# DistilBERT classifier
for max_erase in 0 4 8 12
# for max_erase in 0 10 20 30
do
    python main.py \
        --num_prompts 120 \
        --mode insertion \
        --eval_type safe \
        --max_erase $max_erase \
        --num_adv 1 \
        --use_classifier \
        --safe_prompts data/safe_prompts_test.txt \
        --model_wt_path models/distilbert_insertion.pt
done

# Multiple adversaries
# for num_adv in 1 2
# do
#     for max_erase in 0 2 4 6
#     do
#         python main.py \
#             --num_prompts 30 \
#             --mode insertion \
#             --eval_type safe \
#             --max_erase $max_erase \
#             --num_adv $num_adv \
#             --safe_prompts data/safe_prompts.txt
#     done
# done