#!/bin/bash

# Evaluation script for suffix attacks with max erase running from 0 to 40 with step 10
for max_erase in 0 10 20 30
do
    # python main.py --num_prompts 200 --max_erase $max_erase
    # python main.py --num_prompts 120 --max_erase $max_erase --use_classifier --safe_prompts data/safe_prompts_test.txt
    python main.py --num_prompts 120 --max_erase $max_erase --safe_prompts data/safe_prompts_test.txt
done