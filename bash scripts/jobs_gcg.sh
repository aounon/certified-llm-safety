#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Create a track_gcg.txt file if it doesn't exist
touch logs/track_gcg.txt

# Loop through the range of num_adv values and run gcg.py for each value
for num_iters in 400 200
do
    for num_adv in 120 # 60 80 100 # 0 2 4 6 8 10 12 14 16 18 20
    do
        if grep -Fxq "${num_iters} ${num_adv}" logs/track_gcg.txt
        then
            continue
        fi
        python gcg.py \
            --model_wt_path models/distilbert_suffix.pt \
            --num_adv $num_adv \
            --num_iters $num_iters \
            --save_dir data/suffix_model_${num_iters}

        python gcg.py \
            --model_wt_path models/distilbert_greedy.pt \
            --num_adv $num_adv \
            --num_iters $num_iters \
            --save_dir data/greedy_model_${num_iters}

        echo "${num_iters} ${num_adv}" >> logs/track_gcg.txt
    done
done