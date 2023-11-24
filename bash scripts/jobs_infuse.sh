#!/bin/sh

for max_erase in 0 1 2 3
do
    python main.py --num_prompts 100 --mode infusion --eval_type safe --max_erase $max_erase #--use_classifier
done