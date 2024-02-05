#!/bin/bash

# To evaluate the performance of the gradient-based method on the safe prompts.
# The method should not label safe prompts as harmful. So, percent harmful should be low.

for num_iters in 0 25 50 75 100
do
    python grad_ec.py \
        --prompts_file data/safe_prompts_test.txt \
        --model_wt_path models/distilbert_suffix.pt \
        --num_iters $num_iters \
        --results_file results/grad_ec_safe_suffix.json
done