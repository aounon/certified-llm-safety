#!/bin/sh

# using DistilBERT safety classifier
for max_erase in 0 1 2 3
# for max_erase in 0 2 4 6
do
    python main.py \
        --num_prompts 120 \
        --mode infusion \
        --eval_type safe \
        --max_erase $max_erase \
        --safe_prompts data/safe_prompts_test.txt \
        --use_classifier \
        --model_wt_path models/distilbert_infusion.pt
done

# using Llama 2
# for max_erase in 0 1 2 3
# do
#     python main.py \
#         --num_prompts 100 \
#         --mode infusion \
#         --eval_type safe \
#         --max_erase $max_erase \
#         --safe_prompts data/safe_prompts.txt
# done
