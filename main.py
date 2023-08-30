import torch
import transformers
from transformers import AutoTokenizer

import random

from defenses import erase_and_check, is_harmful, progress_bar

num_prompts = 200

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# # Safe prompts
# print("Safe prompts:")
# # Load prompts from text file
# with open("data/safe_prompts.txt", "r") as f:
#     prompts = f.readlines()
#     prompts = [prompt.strip() for prompt in prompts]

# # Sample a random subset of the prompts
# prompts = random.sample(prompts, num_prompts)

# count_safe = 0
# # Check if the prompts are harmful
# for i in range(num_prompts):
#     prompt = prompts[i]
#     # print(prompt)
#     harmful = erase_and_check(prompt, pipeline, tokenizer)
#     if not harmful:
#         count_safe += 1

#     print("    Checking safety... " + progress_bar((i + 1) / num_prompts) + " done. Detected safe = {percent:4.1f}".format(percent = count_safe / (i + 1) * 100) + "%", end="\r")
#     # print(harmful)

# print("")

# Harmful prompts
print("Harmful prompts:")
# Load prompts from text file
with open("data/harmful_prompts.txt", "r") as f:
    prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]

# Sample a random subset of the prompts
prompts = random.sample(prompts, num_prompts)

count_harmful = 0
# Check if the prompts are harmful
for i in range(num_prompts):
    prompt = prompts[i]
    # print(prompt)
    harmful = is_harmful([prompt], pipeline, tokenizer)
    if harmful[0]:
        count_harmful += 1

    print("    Checking safety... " + progress_bar((i + 1) / num_prompts) + " done. Detected harmful = {percent:4.1f}".format(percent = count_harmful / (i + 1) * 100) + "%", end="\r")
    # print(harmful)

print("")
