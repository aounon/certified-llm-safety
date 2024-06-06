#!/bin/bash

# Evaluation script for suffix attacks with max erase running from 0 to 30 with step 10

# Llama 2
for max_erase in 0 10 20 30
do
    python main.py \
        --num_prompts 520 \
        --mode suffix \
        --eval_type safe \
        --max_erase $max_erase \
        --safe_prompts data/safe_prompts.txt \
        --llm_name 'Llama-2' --results_dir 'results/Llama-2'
done
exit

# Llama 2 13B
# for max_erase in 0 10 20 30
# do
#     python main.py \
#         --num_prompts 520 \
#         --mode suffix \
#         --eval_type safe \
#         --max_erase $max_erase \
#         --safe_prompts data/safe_prompts.txt \
#         --llm_name 'Llama-2-13B' --results_dir 'results/Llama-2-13B'
# done
# exit

# # Llama 3
# for max_erase in 0 10 20 30
# do
#     python main.py \
#         --num_prompts 520 \
#         --mode suffix \
#         --eval_type safe \
#         --max_erase $max_erase \
#         --safe_prompts data/safe_prompts.txt \
#         --llm_name 'Llama-3' --results_dir 'results/Llama-3'
# done
# exit

# GPT
# for max_erase in 0 10 20 30
# do
#     python main.py \
#         --num_prompts 100 \
#         --mode suffix \
#         --eval_type safe \
#         --max_erase $max_erase \
#         --safe_prompts data/safe_prompts.txt \
#         --llm_name 'GPT-3.5' --results_dir 'results/GPT'
# done

# Llama 2
for max_erase in 0 10 20 30
do
    python main.py \
        --num_prompts 520 \
        --mode suffix \
        --eval_type safe \
        --max_erase $max_erase \
        --safe_prompts data/safe_prompts.txt
done

# DistilBERT classifier
for max_erase in 0 10 20 30
do
    python main.py \
        --num_prompts 120 \
        --mode suffix \
        --eval_type safe \
        --max_erase $max_erase \
        --use_classifier \
        --model_wt_path models/distilbert_suffix.pt \
        --safe_prompts data/safe_prompts_test.txt
done