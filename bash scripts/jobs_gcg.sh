#!/bin/bash

# Loop through the range of num_adv values and run gcg.py for each value
for num_adv in 0 2 4 6 8 10 12 14 16 18 20
do
    python gcg.py \
        --model_wt_path models/distilbert_suffix.pt \
        --num_adv $num_adv
done
