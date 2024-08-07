#!/bin/bash
# Script to train a safety classifier on the safety dataset.

for mode in "suffix" "insertion" "infusion"
do
    python safety_classifier.py \
            --safe_train data/safe_prompts_train_${mode}_erased.txt \
            --safe_test data/safe_prompts_test_${mode}_erased.txt \
            --save_path models/distilbert_${mode}.pt
done

# Suffix mode
# python safety_classifier.py \
#     --safe_train data/safe_prompts_train_suffix_erased.txt \
#     --safe_test data/safe_prompts_test_suffix_erased.txt \
#     --save_path models/distilbert_suffix.pt
