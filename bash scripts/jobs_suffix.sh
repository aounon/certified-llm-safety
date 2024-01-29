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