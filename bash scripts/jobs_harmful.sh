#!/bin/sh

# LLM based filters
# python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt --llm_name 'GPT-3.5' --results_dir 'results/GPT'
# python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt --llm_name 'Llama-3' --results_dir 'results/Llama-3'
# python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt --llm_name 'Llama-2-13B' --results_dir 'results/Llama-2-13B'
python main.py --num_prompts 520 --eval_type harmful --harmful_prompts data/harmful_prompts.txt --llm_name 'Llama-2' --results_dir 'results/Llama-2'

exit

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