#!/bin/sh

for num_iters in 0 5 10 25 100
# for num_iters in 0 25 50 75 100
do
    echo "num_iters $num_iters:"
    python main.py \
        --num_prompts 120 \
        --eval_type grad_ec \
        --use_classifier \
        --model_wt_path models/distilbert_suffix.pt \
        --num_iters $num_iters
done