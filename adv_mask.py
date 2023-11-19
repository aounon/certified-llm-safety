## Erase and Check via L0 optimization
## Method: Iterative Hard Thresholding
## argmax_{mask} loss(prompt \odot mask) s.t. mask has length 20 

## Note: masking is performed by simply doing mask * prompt, as 
## [PAD] token is mapping to 0 anyways

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from termcolor import colored
import argparse


def projection(mask, adv_len=10, threshold = 0.5):
    
    # Find bottom k mask values
    _, ind = torch.topk(mask, k = adv_len, largest=False)

    # Find mask values less than a threshold
    ind_thresh = mask > threshold

    new_mask = torch.ones_like(mask)
    new_mask[ind] = 0
    new_mask[ind_thresh] = 1
    return new_mask

def adversarial_mask(input, model, tokenizer, word_embeddings,
        num_adv=10, num_iters=50, init_temp=0.5, reg_const=0.1):
    """
    Find an adversarial mask for the safety classifier 

    Args:
        input: Input prompt
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        word_embeddings: Embedding layer of the safety classifier
        embedding_matrix: Embedding matrix of the safety classifier
        num_adv: Number of adversarial tokens to append
        num_iters: Number of iterations
        batch_size: Batch size
    Returns:
        adv_prompt: Input prompt with adversarial suffix
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize adversarial sequence and convert to tokens
    tokens = torch.tensor(tokenizer.encode(input + ('!' * num_adv))).unsqueeze(0).to(device)
    prompt_len = len(tokenizer.decode(torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(device)[0][1:-1]))

    # Heuristic: Initialize to a probability value such that 
    # about num_adv tokens are erased on average
    mask_probs = torch.ones_like(tokens).float() * (1 - (num_adv / prompt_len))
    mask_probs = torch.nn.Parameter(mask_probs.requires_grad_().to(device))
    optimizer = torch.optim.SGD(params=mask_probs, lr=1e-3)


    for i in range(num_iters):

        optimizer.zero_grad()

        # Heuristic: Decay temperature over time
        # as temp -> 0; RelaxedBernoulli -> Bernoulli
        temp = init_temp / (i+1)

        # RelaxedBernoulli distribution
        binary_mask = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=temp, probs=torch.sigmoid(mask_probs))

        # Candidate token to test
        sample_tokens = (binary_mask.sample() * tokens).long()
        
        # Regularize mask such that most of the probabilities are close to one
        regularizer = (torch.sigmoid(mask_probs) - 1).pow(2).mean()

        # Get token embeddings
        embeddings = word_embeddings(sample_tokens)
        # Compute gradient of loss w.r.t. embeddings
        embeddings.retain_grad()
        output = model(inputs_embeds=embeddings, labels=torch.tensor([0]).to(device)) # Class 0 is harmful

        loss = output.loss + reg_const * regularizer
        loss.backward() 
        optimizer.step()

        if i % 5 == 0:
            print("Iteration", i, " Loss: ",  loss.item(), 
                  " Output: ", output.loss.item(), 
                  " Reg: ", regularizer.item())

            sampled_prompt = tokenizer.decode(sample_tokens[0][1:-1])
            print("Sampled Prompt: " + sampled_prompt)
    
    return sampled_prompt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adversarial prompts using GCG.')
    parser.add_argument('--num_adv', type=int, default=10, help='Number of adversarial tokens to append')
    parser.add_argument('--prompts_file', type=str, default='data/harmful_prompts_test.txt', help='File containing prompts')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model_wt_path = 'models/distillbert_saved_weights.pt'
    model.load_state_dict(torch.load(model_wt_path))
    model.to(device)
    model.eval()

    num_adv = args.num_adv
    prompts_file = args.prompts_file

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")

    # Open file to write prompts
    for input in prompts:
        print("ORG PROMPT: " + input)

        sampled_prompt = adversarial_mask(input, model, tokenizer, 
                                      model.distilbert.embeddings.word_embeddings,num_adv=num_adv,num_iters=100,
                                      init_temp=0.5, reg_const=0.1)

        
        print("SAMPLED ADV PROMPT: " + sampled_prompt)
        tokens = torch.tensor(tokenizer.encode(sampled_prompt)).unsqueeze(0).to(device)
        model_output = model(tokens)
        print("Prediction: " + ("safe" if model_output[0].argmax().item() == 1 else "harmful"))
        print()
