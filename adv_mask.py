## Erase and Check via approximate L0 optimization
## argmax_{mask} loss(prompt \odot mask) s.t. mak has least possible sparsity 

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse

def adversarial_mask(input, model, tokenizer, word_embeddings,
        num_iters=50, init_temp=0.5, reg_const=0.1):
    """
    Find an adversarial mask for the safety classifier 

    Args:
        input: Input prompt
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        word_embeddings: Embedding layer of the safety classifier
        num_iters: Number of iterations
        init_temp: Initial temperature value for RelaxedBernoulli
        reg_const: Regularization constant enforcing masks to have small sparsity
    Returns:
        sampled_prompt: input prompt with learnt mask 
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize adversarial sequence and convert to tokens
    tokens = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(device)

    # Initialize to all ones
    mask_probs = torch.zeros_like(tokens).float()
    mask_probs = torch.nn.Parameter(mask_probs.requires_grad_().to(device))
    optimizer = torch.optim.SGD(params=[mask_probs], lr=1e-1, momentum=0.9, weight_decay=0.)

    apply_mask_to_tokens = False
    print_outputs = False

    for i in range(num_iters):

        optimizer.zero_grad()

        # Heuristic: Decay temperature over time
        # as temp -> 0; mask becomes binary
        temp = init_temp / (i+1)

        sampled_mask = torch.sigmoid(mask_probs / temp)

        # Regularize mask such that most of the probabilities are close to one
        regularizer = (torch.sigmoid(mask_probs) - 1).pow(2).mean()

        if apply_mask_to_tokens:
            sampled_tokens = (sampled_mask * tokens).long()
            embeddings = word_embeddings(sampled_tokens)
        else:
            embeddings = word_embeddings(tokens)
            embeddings = sampled_mask.unsqueeze(2) * embeddings
        
        # Class 0 is harmful
        output = model(inputs_embeds=embeddings, labels=torch.tensor([0]).to(device)) 

        loss = output.loss + reg_const * regularizer
        loss.backward() 
        optimizer.step()

        if apply_mask_to_tokens is False:
            sampled_tokens = torch.round(sampled_mask * tokens).long()
            sampled_prompt = tokenizer.decode((sampled_tokens)[0][1:-1])

        if i % 10 == 0 and print_outputs:
            print("Iteration", i, " Loss: ",  loss.item(), 
                  " Output: ", output.loss.item(), 
                  " Reg: ", regularizer.item())
             
            print("Sampled Prompt: " + sampled_prompt)
            print("Prediction: " + ("safe" if output.logits[0].argmax().item() == 1 else "harmful"))
            print(output.logits[0])
            print()

    
    return sampled_prompt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
    parser.add_argument('--prompts_file', type=str, default='data/adversarial_prompts_t_21.txt', help='File containing prompts')
    parser.add_argument('--num_iters', type=int, default=200, help='Number of iterations')


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

    prompts_file = args.prompts_file
    num_iters = args.num_iters

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")

    # Open file to write prompts
    for input in prompts:
        #print("ORG PROMPT: " + input)
        tokens = torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(device)
        model_output = model(tokens)
        print("Original Prediction: " + ("safe" if model_output[0].argmax().item() == 1 else "harmful"))

        word_embeddings = model.distilbert.embeddings.word_embeddings

        sampled_prompt = adversarial_mask(input, model, tokenizer, 
                                      word_embeddings,
                                      num_iters=num_iters,
                                      init_temp=1., reg_const=1e-4)
 
        #print("MASKED PROMPT: " + sampled_prompt)
        tokens = torch.tensor(tokenizer.encode(sampled_prompt)).unsqueeze(0).to(device)
        model_output = model(tokens)
        print("Masked Prediction: " + ("safe" if model_output[0].argmax().item() == 1 else "harmful"))
        #print(model_output[0])
        print()
