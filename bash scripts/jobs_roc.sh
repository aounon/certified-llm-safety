#!/bin/sh

for max_erase in 20 40 60 80 100 120
do
    python main.py \
        --num_prompts 120 \
        --eval_type roc_curve \
        --use_classifier \
        --model_wt_path models/distilbert_suffix.pt \
        --max_erase $max_erase \
        --ec_variant RandEC \
        --adv_prompts_dir data/suffix_model \
        --results_dir results/roc

    python main.py \
        --num_prompts 120 \
        --eval_type roc_curve \
        --use_classifier \
        --model_wt_path models/distilbert_greedy.pt \
        --max_erase $max_erase \
        --ec_variant GreedyEC \
        --adv_prompts_dir data/greedy_model \
        --results_dir results/roc

    python main.py \
        --num_prompts 120 \
        --eval_type roc_curve \
        --use_classifier \
        --model_wt_path models/distilbert_suffix.pt \
        --max_erase $max_erase \
        --ec_variant GradEC \
        --adv_prompts_dir data/suffix_model \
        --results_dir results/roc

    python main.py \
        --num_prompts 120 \
        --eval_type roc_curve \
        --use_classifier \
        --model_wt_path models/distilbert_greedy.pt \
        --max_erase $max_erase \
        --ec_variant GreedyGradEC \
        --adv_prompts_dir data/greedy_model \
        --results_dir results/roc
done