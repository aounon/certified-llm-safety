for num_adv in 2
do
    for max_erase in 0 2 4 6
    do
        python main.py --num_prompts 30 --mode insertion --max_erase $max_erase --num_adv $num_adv
    done
done