#!/bin/sh

max_erase=20
nvidia-smi
python -u main.py --append-adv --num_prompts 200 --eval_type harmful --max_erase 20
python -u main.py --num_prompts 200 --eval_type harmful --max_erase 20
python -u main.py --num_prompts 200 --eval_type safe --max_erase 20