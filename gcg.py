import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import random

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

top_k = 10
batch_size = 100
num_iters = 20 # Number of iterations of GCG
num_adv = 10 # Number of adversarial tokens to append

# Load prompts
prompts = []
with open('data/harmful_prompts_test.txt', 'r') as f:
    for line in f:
        prompts.append(line.strip())

print("Loaded " + str(len(prompts)) + " prompts.")

# Open file to write prompts
f = open('data/adversarial_prompts_t_' + str(num_adv) + '.txt', 'w')
for input in prompts:
    print("ORG PROMPT: " + input)

    adv_input = input + ('!' * num_adv)
    tokens = torch.tensor(tokenizer.encode(adv_input)).unsqueeze(0).to(device)

    for _ in range(num_iters):
        # Get token embeddings
        embeddings = model.distilbert.embeddings.word_embeddings(tokens)

        # Compute gradient of loss w.r.t. embeddings
        embeddings.retain_grad()
        output = model(inputs_embeds=embeddings, labels=torch.tensor([1]).to(device)) # Class 1 is safe
        loss = output.loss
        (-loss).backward()  # Minimize loss
        gradients = embeddings.grad

        # Get embeding matrix
        embedding_matrix = model.distilbert.embeddings.word_embeddings.weight

        # Dot product of gradients and embedding matrix
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)[-num_adv-1:-1]

        # Create a batch of adversarial prompts by uniformly sampling from top k adversarial tokens
        tokens_batch = [tokens.clone().detach()]
        for _ in range(batch_size):
            random_indices = torch.randint(0, top_k, (num_adv,)).to(device)
            selected_tokens = torch.gather(top_k_adv, 1, random_indices.unsqueeze(1))
            batch_item = tokens.clone().detach()
            batch_item[0, -num_adv-1:-1] = selected_tokens.squeeze(1)
            tokens_batch.append(batch_item)

        tokens_batch = torch.cat(tokens_batch, dim=0)

        # Pick batch elements with highest logit
        output = model(inputs_embeds=model.distilbert.embeddings.word_embeddings(tokens_batch))
        max_idx = torch.argmax(output.logits[:, 1])
        tokens = tokens_batch[max_idx].unsqueeze(0)

    print("ADV PROMPT: " + tokenizer.decode(tokens[0][1:-1]))
    print("Prediction: " + ("safe" if torch.argmax(output.logits[max_idx]).item() == 1 else "harmful"))
    f.write(tokenizer.decode(tokens[0][1:-1]) + '\n')
    f.flush()

f.close()