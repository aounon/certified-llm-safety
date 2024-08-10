# Implements the greedy version of the erase-and-check procedure

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse, time, json, os

from defenses import progress_bar

def grad_ec_subroutine(tokens, model, word_embeddings, device, num_iters=10):
    """
    This function implements GradEC -- a gradient based approximation of the erase-and-check procedure.
    It optimizes a mask over the input prompt to erase tokens and check if the resulting
    prompt is harmful or not. If the resulting sequence is harmful, it returns True, else False. It also
    returns the erased prompt.

    Args:
        tokens: Input token
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        word_embeddings: Embedding layer of the safety classifier
        num_iters: Number of iterations
    Returns:
        harmful: Whether the input prompt is harmful or not
        masked_prompt: input prompt with some tokens masked out 
    """

    # Initialize to all zeros
    mask_logits = torch.ones_like(tokens).float()
    mask_logits = torch.nn.Parameter(mask_logits.requires_grad_().to(device))
    optimizer = torch.optim.SGD(params=[mask_logits], lr=10.0, momentum=0.0, weight_decay=1e-2)

    for i in range(num_iters):

        optimizer.zero_grad()

        mask_sigmoid = torch.sigmoid(mask_logits)

        embeddings = word_embeddings(tokens)
        embeddings = mask_sigmoid.unsqueeze(2) * embeddings +  (1 -  mask_sigmoid.unsqueeze(2)) * word_embeddings(0 * tokens)
        
        # Class 0 is harmful
        output = model(inputs_embeds=embeddings, labels=torch.tensor([0]).to(device)) 

        loss = output.loss
        loss.backward() 
        optimizer.step()

    erase_index = mask_sigmoid.argmin()
    return torch.cat((tokens[:, :erase_index], tokens[:, erase_index+1:]), dim=1)

def greedy_grad_ec(prompt, model, tokenizer, word_embeddings,
                   num_iters=20, output_subsequence=False):
    '''
    Implements the greedy version of the erase-and-check procedure using gradient-based 
    optimization. At each step, the algorithm erases a token from the current prompt
    such that the softmax score for the harmful class is maximized, identified via optimization.
    It repeats this process until the safety classifier labels the prompt as harmful. If the
    model does not label the prompt as harmful, the algorithm declares
    the original prompt as safe.

    Args:
        prompt: Input prompt (string)
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        num_iters: Number of iterations

    Returns:
        True if the prompt is harmful, False otherwise
    '''

    device = model.device

    # Tokenize prompt
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    # Get initial model output
    model_output = model(tokens)
    output_class = model_output[0].argmax().item()

    # If the prompt is initially labelled harmful, return True
    # Class 1: Safe, Class 0: Harmful
    if output_class == 0:
        if output_subsequence:
            return True, prompt
        return True
    
    # Otherwise, erase tokens until the prompt is labelled harmful
    for _ in range(num_iters):
        if tokens.shape[1] < 3:     # Loop below will not work
            if output_subsequence:
                return False, prompt
            return False

        erased_subsequence = grad_ec_subroutine(tokens, model, word_embeddings, device=device, num_iters=2) 
        model_output = model(erased_subsequence)
        output_class = model_output[0].argmax().item()

        # If the max score is greater than threshold, return True
        if output_class == 0:
            # Decode the prompt and print it
            # decoded_prompt = tokenizer.decode(tokens_batch[max_index])
            # print("Original prompt:\t", prompt)
            # print("Erased prompt:\t", decoded_prompt)
            # input("Press Enter to continue...")
            if output_subsequence:
                return True, tokenizer.decode(erased_subsequence, skip_special_tokens=True)
            return True
        
        # Otherwise, update tokens
        tokens = erased_subsequence

    if output_subsequence:
        return False, prompt
    return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
    parser.add_argument('--prompts_file', type=str, default='data/adversarial_prompts_t_20.txt', help='File containing prompts')
    parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations')
    parser.add_argument('--model_wt_path', type=str, default='models/distilbert_suffix.pt', help='Path to model weights')
    parser.add_argument('--results_file', type=str, default='results/greedy_ec_results.json', help='File to store results')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model_wt_path = args.model_wt_path
    
    model.load_state_dict(torch.load(model_wt_path, map_location=device))
    model.to(device)
    model.eval()

    prompts_file = args.prompts_file
    num_iters = args.num_iters
    results_file = args.results_file

    print('\n* * * * * * * Experiment Details * * * * * * *')
    print('Prompts file:\t', prompts_file)
    print('Iterations:\t', str(num_iters))
    print('Model weights:\t', model_wt_path)
    print('* * * * * * * * * * * ** * * * * * * * * * * *\n')

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")
    list_of_bools = []
    start_time = time.time()

    # Open results file and load previous results JSON as a dictionary
    results_dict = {}
    # Create results file if it does not exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
    with open(results_file, 'r') as f:
        results_dict = json.load(f)

    for num_done, input_prompt in enumerate(prompts):
        decision = greedy_grad_ec(input_prompt, model, tokenizer, 
                                  word_embeddings=model.distilbert.embeddings.word_embeddings, 
                                  num_iters=num_iters)
        list_of_bools.append(decision)

        percent_harmful = (sum(list_of_bools) / len(list_of_bools)) * 100.
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (num_done + 1)

        print("  Checking safety... " + progress_bar((num_done + 1) / len(prompts)) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r")
        
    print("")

    # Save results
    results_dict[str(dict(num_iters = num_iters))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt)
    print("Saving results to", results_file)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
