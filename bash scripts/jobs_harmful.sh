#!/bin/sh

# Llama 2
python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt

# DistilBERT
for mode in "suffix" "insertion" "infusion"
do
    python main.py \
            --num_prompts 120 \
            --eval_type harmful \
            --use_classifier --model_wt_path models/distilbert_${mode}.pt \
            --harmful_prompts data/harmful_prompts_test.txt
done

# Confirm Llama 2 results with more runs
for i in {1..10}
do
    echo "Run $i:"
    python main.py \
            --eval_type harmful \
            --harmful_prompts data/harmful_prompts.txt \
            --num_prompts 1000
done