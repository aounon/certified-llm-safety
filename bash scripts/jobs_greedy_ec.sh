#!/bin/sh

for num_iters in 0 2 4 6 8
do
    echo "num_iters $num_iters:"
    python main.py \
        --num_prompts 120 \
        --eval_type greedy_ec \
        --use_classifier \
        --model_wt_path models/distilbert_suffix.pt \
        --num_iters $num_iters
done