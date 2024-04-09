#!/bin/sh

for ratio in 1.0        # 0.0 0.1 0.2 0.3 0.4
do
    echo "Sampling Ratio $ratio:"
    python main.py \
        --num_prompts 120 \
        --eval_type empirical \
        --mode suffix \
        --max_erase 20 \
        --use_classifier \
        --model_wt_path models/distilbert_suffix.pt \
        --randomize --sampling_ratio $ratio
done