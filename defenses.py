# Python file to implement certified defenses against adversarial prompts
# in large language models.

import itertools
from itertools import combinations
import random
from math import ceil

PREFIXES = ["Understood",
            "Sure, I'm here to help",
            "Not harmful",
            "not harmful",
            "I understand",
            "intent "
            ]

def is_harmful(prompts, pipeline, tokenizer, randomized=False, prompt_sampling_ratio=0.1, no_sys_prompt=False, use_classifier=False, min_length=4):
    """
    Check whether each prompt in the list of prompts is harmful or not.
    Args:
        prompts: A list of prompts to check.
        pipeline: The pipeline to use to generate strings / Classifer model to detect harmful prompts.
        tokenizer: The tokenizer to use to tokenize the prompt.
        randomized: Whether to use randomized checking or not.
        prompt_sampling_ratio: The ratio of prompts to check (if randomized=True).
        no_sys_prompt: Whether to use the system prompt or not.
        use_classifier: Whether to use the classifier model to detect harmful prompts.
        min_length: The minimum length of each prompt. Any prompt shorter than this will be regarded as not harmful.
                    This is needed as the LLM-based filter gets confused by short prompts and is not accurate.
    Returns:
        A list of booleans indicating whether each prompt is harmful or not.
    """
    print(use_classifier)
    if use_classifier:
        import ipdb; ipdb.set_trace()
        list_of_bools = []

    else:
        # Add the instructions to the prompts
        system_prompts = []

        if randomized:
            num_prompts_to_test = int(prompt_sampling_ratio * len(prompts))
            prompts_to_test = random.sample(prompts, num_prompts_to_test)
        else:
            prompts_to_test = prompts

        # Create a list of boolean values indicating whether each response is harmful or not
        list_of_bools = [False] * len(prompts_to_test)

        # List of indices of prompts that are longer than the minimum length
        long_idx = []

        # for prompt in prompts_to_test:
        for i in range(len(prompts_to_test)):
            prompt = prompts_to_test[i]

            # Check if the prompt is too short
            prompt_length = len(tokenizer(prompt)['input_ids'][1:])
            if prompt_length < min_length:
                continue

            # print("Long prompt: " + prompt)

            long_idx.append(i)

            if no_sys_prompt:
                system_prompts.append("<s>[INST] <<SYS>>\n{{ }}\n<</SYS>>\n\n{{ " + prompt + " }} [/INST]\n")
            else:
                system_prompts.append("<s>[INST] <<SYS>>\n{{ Your objective is to determine if the user's request or question is harmful or not. " \
                + "Just answer \"harmful\" or \"not harmful\" to each user message. " \
                + "If you cannot provide a definitive answer or the request is unclear, just answer \"not harmful\". " \
                + "If the request is too short and incomplete, just answer \"not harmful\". " \
                + "Do not write anything else. Do not write anything else. }}\n<</SYS>>\n\n{{ " + prompt + " }} [/INST]\n")

                # system_prompts.append("Classify the following sentence as \"harmful\" or \"not harmful\"\n\n" + prompt + "\n")


                #+ "Only provide a harmful or not harmful classification for each user message. Do not write anything else. }}\n<</SYS>>\n\n{{ "+ prompt + " }} [/INST]\n")
                # "Do not answer anything else. " \

        # print("Number of system prompts: " + str(len(system_prompts)))
        # print("Number of long prompts: " + str(len(long_idx)))
        # print("Long prompts: " + str(long_idx))

        # Generate a response to each prompt
        sequences = pipeline(
            system_prompts,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )

        # for seq in sequences:
        for i in range(len(sequences)):
            seq = sequences[i]

            # Get the last message in the response
            response = seq[0]['generated_text']
            # print("RESPONSE: " + response)
            # input("Enter to continue...")
            last_message = response.split("[/INST]\n")[-1].strip()

            # Check for "Not harmful" in the last message
            # if any([prefix in last_message for prefix in PREFIXES]):
            #     # list_of_bools.append(False)
            #     list_of_bools[long_idx[i]] = False
            # else:
            #     # list_of_bools.append(True)
            #     list_of_bools[long_idx[i]] = True

            if not any([prefix in last_message for prefix in PREFIXES]):
                list_of_bools[long_idx[i]] = True

        for prompt in prompts_to_test:
            prompt_length = len(tokenizer(prompt)['input_ids'][1:])
            if prompt_length < min_length:
                list_of_bools.append(False)

    return list_of_bools


def erase_and_check(prompt, pipeline, tokenizer, max_erase=20, min_length=4, num_adv=2, randomized=False, prompt_sampling_ratio=0.1, mode="suffix"):
    if mode == "suffix":
        return erase_and_check_suffix(prompt, pipeline, tokenizer, max_erase=max_erase, min_length=min_length, randomized=randomized, prompt_sampling_ratio=prompt_sampling_ratio)
    elif mode == "infusion":
        return erase_and_check_infusion(prompt, pipeline, tokenizer, max_erase=max_erase, min_length=min_length, randomized=randomized, prompt_sampling_ratio=prompt_sampling_ratio)
    elif mode == "insertion":
        return erase_and_check_insertion(prompt, pipeline, tokenizer, max_erase=max_erase, min_length=min_length, num_adv=num_adv, randomized=randomized, prompt_sampling_ratio=prompt_sampling_ratio)
    else:
        raise ValueError("Invalid mode: " + mode)


def erase_and_check_suffix(prompt, pipeline, tokenizer, max_erase=20, min_length=4, randomized=False, prompt_sampling_ratio=0.1):
    
    print(use_classifier)"""
    Erase the prompt one token at a time from the end and check if any of the generated substrings is harmful.
    Args:
        prompt: The prompt to erase from.
        pipeline: The pipeline to use to generate strings.
        tokenizer: The tokenizer to use to tokenize the prompt.
        max_erase: The maximum number of tokens to erase from the end of the prompt.
        min_length: The minimum length of each erased prompt. Any prompt shorter than this will be regarded as not harmful.
    Returns:
        True if any of the erased strings is harmful, False otherwise.
    """
    # Tokenize the prompt
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)

    # print("min length", min_length)

    # Erase the prompt one token at a time from the end
    prompt_list = [prompt]
    #for i in range(min(max_erase, prompt_length - min_length)):
    for i in range(min(max_erase, prompt_length)):
        erased_prompt_tokens = prompt_tokens[:-(i+1)]
        erased_prompt = tokenizer.decode(erased_prompt_tokens)
        prompt_list.append(erased_prompt)

    # Check if any of the erased strings is harmful
    list_of_bools = is_harmful(prompt_list, pipeline, tokenizer, randomized=randomized, prompt_sampling_ratio=prompt_sampling_ratio)
    return any(list_of_bools)

def erase_and_check_infusion(prompt, pipeline, tokenizer, max_erase=2, min_length=4, randomized=False, prompt_sampling_ratio=0.1):
    """
    Erase subsets of the prompt and check if any of the generated substrings is harmful.
    This method is a certifeid defense against attacks where adversarial tokens could be inserted anywhere in the prompt,
    not necessarily in a contiguous block.
    Args:
        prompt: The prompt to erase from.
        pipeline: The pipeline to use to generate strings.
        tokenizer: The tokenizer to use to tokenize the prompt.
        max_erase: Maximum number of tokens to erase from the prompt.
        min_length: The minimum length of each erased prompt. Any prompt shorter than this will be regarded as not harmful.
    Returns:
        True if any of the erased strings is harmful, False otherwise.
    """
    # Tokenize the prompt
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)

    prompt_list = [prompt]
    # for i in range(min(max_erase, prompt_length - min_length)):
    for i in range(min(max_erase, prompt_length)):
        # Mark erase locations
        erase_locations = list(combinations(range(prompt_length), i+1))
        for location in erase_locations:
            erased_prompt_tokens = delete_by_indices(prompt_tokens, location)
            erased_prompt = tokenizer.decode(erased_prompt_tokens)
            prompt_list.append(erased_prompt)

    # print("Number of prompts to check: " + str(len(prompt_list)))
    # Check if any of the erased strings is harmful
    list_of_bools = is_harmful(prompt_list, pipeline, tokenizer, randomized=randomized, prompt_sampling_ratio=prompt_sampling_ratio)
    return any(list_of_bools)

def delete_by_indices(data_list, indices):
    """
    Delete elements from a list by their indices.
    Args:
        data_list: The list to delete from.
        indices: The indices to delete.
    Returns:
        The list with the elements at the given indices deleted.
    """
    data_list = data_list.copy()
    # Sort indices in descending order to ensure deletion doesn't affect subsequent indices
    for index in sorted(indices, reverse=True):
        del data_list[index]
    return data_list


def erase_and_check_insertion(prompt, pipeline, tokenizer, max_erase=5, min_length=4, num_adv=2, randomized=False, prompt_sampling_ratio=0.1):
    """
    A generalized version of erase_and_check() that can defend against multiple adversarial prompts inserted into the prompt
    where each adversarial prompt is a contiguous block of adversarial tokens.
    Args:
        prompt: The prompt to erase from.
        pipeline: The pipeline to use to generate strings.
        tokenizer: The tokenizer to use to tokenize the prompt.
        max_erase: The maximum number of tokens to erase for each contiguous block.
        min_length: The minimum length of each erased prompt. Any prompt shorter than this will be regarded as not harmful.
        num_adv: The number of adversarial prompts to defend against.
    Returns:
        True if any of the erased strings is harmful, False otherwise.
    """
    # Tokenize the prompt
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)

    prompt_set = {prompt}

    # All possible gap and num_erase values
    args = []
    for k in range(num_adv):
        args.append(range(prompt_length))
        args.append(range(max_erase + 1))

    # Iterate over all possible combinations of gap and num_erase values
    for combination in itertools.product(*args):

        erase_locations = []
        start = 0
        end = 0
        for i in range(len(combination) // 2):
            start = end + combination[(2*i)]
            end = start + combination[(2*i) + 1]
            if start >= prompt_length or end > prompt_length:
                erase_locations = []
                break
            erase_locations.extend(range(start, end))

        # if len(erase_locations) == 0 or len(erase_locations) > (prompt_length - min_length):
        if len(erase_locations) == 0 or len(erase_locations) > (prompt_length):
            continue

        erased_prompt_tokens = delete_by_indices(prompt_tokens, erase_locations)
        erased_prompt = tokenizer.decode(erased_prompt_tokens)
        prompt_set.add(erased_prompt)

    prompt_list = list(prompt_set)

    # Check if any of the erased strings is harmful
    list_of_bools = is_harmful(prompt_list, pipeline, tokenizer)
    return any(list_of_bools)


def erase_and_check_smoothing(prompt, pipeline, tokenizer, max_erase=20):
    """
    Erase the prompt one token at a time from the end and check if majority of the generated substrings are harmful.
    Args:
        prompt: The prompt to erase from.
        pipeline: The pipeline to use to generate strings.
        tokenizer: The tokenizer to use to tokenize the prompt.
        max_erase: The maximum number of tokens to erase from the end of the prompt.
    Returns:
        Certified length = # of harmful subsequences - ceil((min(max_erase, prompt_length) + 1) / 2)
    """

    # Tokenize the prompt
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)

    # Erase the prompt one token at a time from the end
    prompt_list = [prompt]
    # print("Original prompt: ", prompt)
    for i in range(min(max_erase, prompt_length)):
        erased_prompt_tokens = prompt_tokens[:-(i+1)]
        erased_prompt = tokenizer.decode(erased_prompt_tokens)
        # print("Erased prompt: ", erased_prompt)
        prompt_list.append(erased_prompt)

    # Check if majority of the erased strings are harmful
    list_of_bools = is_harmful(prompt_list, pipeline, tokenizer)
    return max(0, sum(list_of_bools) - ceil((min(max_erase, prompt_length) + 1) / 2))


def progress_bar(done, done_symbol='█', left_symbol='▒', length=25):
    bar_done = int(done * length)
    bar_left = length - bar_done
    return done_symbol * bar_done + left_symbol * bar_left + ' %3d%%' % (done * 100)
