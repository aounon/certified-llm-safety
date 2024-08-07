import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from termcolor import colored
import argparse
import os

def gcg_suffix(input, model, tokenizer, word_embeddings, embedding_matrix,
               num_adv=10, num_iters=50, top_k=50, batch_size=500):
    """
    Generates adversarial suffix for the safety classifier using GCG.
    Args:
        input: Input prompt
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        word_embeddings: Embedding layer of the safety classifier
        embedding_matrix: Embedding matrix of the safety classifier
        num_adv: Number of adversarial tokens to append
        num_iters: Number of iterations of GCG
        top_k: Number of top adversarial tokens to consider
        batch_size: Batch size for GCG
    Returns:
        adv_prompt: Input prompt with adversarial suffix
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if num_adv == 0:
        return tokenizer.decode(tokenizer.encode(input, add_special_tokens=False))

    # Initialize adversarial sequence and convert to tokens
    tokens = torch.tensor(tokenizer.encode(input + ('*' * num_adv))).unsqueeze(0).to(device)
    prompt_len = len(tokenizer.decode(torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(device)[0][1:-1]))

    for iter in range(num_iters):
        # Get token embeddings
        embeddings = word_embeddings(tokens)

        # Compute gradient of loss w.r.t. embeddings
        embeddings.retain_grad()
        output = model(inputs_embeds=embeddings, labels=torch.tensor([1]).to(device)) # Class 1 is safe
        loss = output.loss
        (-loss).backward()  # Minimize loss
        gradients = embeddings.grad

        # Dot product of gradients and embedding matrix
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        # Set dot product of [CLS] and [SEP] tokens to -inf
        cls_token_idx = tokenizer.encode('[CLS]')[1]
        sep_token_idx = tokenizer.encode('[SEP]')[1]
        dot_prod[:, cls_token_idx] = -float('inf')
        dot_prod[:, sep_token_idx] = -float('inf')

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)[-num_adv-1:-1]    # Last token is [SEP]
        
        # # Create a batch of adversarial prompts by uniformly sampling from top k adversarial tokens
        # tokens_batch = [tokens.clone().detach()]
        # for _ in range(batch_size):
        #     random_indices = torch.randint(0, top_k, (num_adv,)).to(device)
        #     selected_tokens = torch.gather(top_k_adv, 1, random_indices.unsqueeze(1))
        #     batch_item = tokens.clone().detach()
        #     batch_item[0, -num_adv-1:-1] = selected_tokens.squeeze(1)
        #     tokens_batch.append(batch_item)

        # Create a batch of adversarial prompts by replacing a random adversarial
        # token with a random top k adversarial token
        tokens_batch = []
        for _ in range(batch_size):
            random_idx = torch.randint(0, num_adv, (1,)).to(device)
            random_top_k_idx = torch.randint(0, top_k, (1,)).to(device)
            batch_item = tokens.clone().detach()
            batch_item[0, -num_adv-1:-1][random_idx] = top_k_adv[random_idx, random_top_k_idx]
            tokens_batch.append(batch_item)

        tokens_batch = torch.cat(tokens_batch, dim=0)

        # Pick batch elements with highest softmax probability for class 1
        output = model(inputs_embeds=word_embeddings(tokens_batch))
        output_softmax = torch.softmax(output.logits, dim=1)
        max_softmax, max_idx = torch.max(output_softmax[:, 1], dim=0)
        # max_logit, max_idx = torch.max(output.logits[:, 1], dim=0)
        if iter == 0 or max_softmax > prev_max:
            tokens = tokens_batch[max_idx].unsqueeze(0)
            prev_max = max_softmax

        adv_prompt = tokenizer.decode(tokens[0][1:-1])
        # print("ADV PROMPT: " + adv_prompt[0:prompt_len] + colored(adv_prompt[prompt_len:], 'red'), end='\r')
    
    # print()

    return adv_prompt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adversarial prompts using GCG.')
    parser.add_argument('--num_adv', type=int, default=10, help='Number of adversarial tokens to append')
    parser.add_argument('--prompts_file', type=str, default='data/harmful_prompts_test.txt', help='File containing prompts')
    parser.add_argument('--model_wt_path', type=str, default='models/distilbert_suffix.pt', help='Path to model weights')
    parser.add_argument('--num_iters', type=int, default=50, help='Number of iterations of GCG')
    parser.add_argument('--save_dir', type=str, default='data', help='Directory to save adversarial prompts')

    args = parser.parse_args()

    model_wt_path = args.model_wt_path
    num_adv = args.num_adv
    num_iters = args.num_iters
    prompts_file = args.prompts_file
    save_dir = args.save_dir

    print("\n* * * * * Experiment Settings * * * * *")
    print("Adversarial tokens: " + str(num_adv))
    print("Number of iterations: " + str(num_iters))
    print("Prompts file: " + prompts_file)
    print("Model weights path: " + model_wt_path)
    print("Save directory: " + save_dir)
    print("* * * * * * * * * * * * * * * * * * * *\n", flush=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model.load_state_dict(torch.load(model_wt_path))
    model.to(device)
    model.eval()

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.", flush=True)

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open file to write prompts
    f = open(f'{save_dir}/adversarial_prompts_t_' + str(num_adv) + '.txt', 'w')
    for input in prompts:
        # print("ORG PROMPT: " + input)

        adv_prompt = gcg_suffix(input, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                                model.distilbert.embeddings.word_embeddings.weight,
                                num_adv=num_adv, num_iters=num_iters)

        
        # print("ADV PROMPT: " + adv_prompt)
        tokens = torch.tensor(tokenizer.encode(adv_prompt)).unsqueeze(0).to(device)
        model_output = model(tokens)
        # print("Prediction: " + ("safe" if model_output[0].argmax().item() == 1 else "harmful"))
        # print()
        f.write(adv_prompt + '\n')
        f.flush()

    f.close()
