# Train the DistilBERT classifier for the greedy approach

import numpy as np

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW

import argparse

from greedy_ec import greedy_ec

def batch_greedy_ec_train(batch, labels, model, tokenizer, num_iters, threshold):
    '''
    For every safe prompt in the batch, append the subsequence 
    that the greedy algorithm identifies as harmful to the batch. If no subsequences are harmful,
    keep the original prompt. For harmful prompts, keep the original prompt.
    '''

    batch_len = len(batch)
    for i in range(batch_len):
        prompt = batch[i]
        # Check if the prompt is safe
        if labels[i] == 1:
            harmful, subsequence = greedy_ec(prompt, model, tokenizer,
                                             num_iters=num_iters,
                                             threshold=threshold,
                                             output_subsequence=True)
            if harmful:
                # print(f"{i}: {subsequence}")
                batch.append(subsequence)
                labels.append(1)

    return batch, labels

def batch_greedy_ec_train_old(batch, labels, model, tokenizer, num_iters):
    '''
    For every safe prompt in the batch, append the subsequence 
    that the greedy algorithm identifies as harmful to the batch. If no subsequences are harmful,
    keep the original prompt. For harmful prompts, keep the original prompt.
    '''

    batch_len = len(batch)
    for i in range(batch_len):
        prompt = batch[i]
        # Check if the prompt is safe
        if labels[i] == 1:
            harmful, subsequence = greedy_ec(prompt, model, tokenizer, num_iters, output_subsequence=True)
            if harmful:
                # print(f"{i}: {subsequence}")
                batch.append(subsequence)
            else:
                # Erase a random number of tokens from the prompt
                tokenized_prompt = tokenizer(prompt, add_special_tokens=False)['input_ids']
                num_erase = np.random.randint(0, len(tokenized_prompt))
                idx = np.random.permutation(len(tokenized_prompt))
                idx = idx[:num_erase]
                erased_prompt = [tokenized_prompt[i] for i in range(len(tokenized_prompt)) if i not in idx]
                erased_prompt = tokenizer.decode(erased_prompt)

                batch.append(erased_prompt)
            labels.append(1)
        else:
            # Balance the batch
            batch.append(prompt)
            labels.append(0)

    return batch, labels

def batch_greedy_ec_test(batch, labels, model, tokenizer, num_iters):
    '''
    For every safe prompt in the batch, append the subsequence 
    that the greedy algorithm identifies as harmful to the batch. If no subsequences are harmful,
    keep the original prompt. For harmful prompts, keep the original prompt.
    '''

    # Initialize the new batch
    new_batch = []

    for i, prompt in enumerate(batch):
        # Check if the prompt is safe
        if labels[i] == 1:
            harmful, subsequence = greedy_ec(prompt, model, tokenizer,
                                             num_iters=num_iters,
                                             output_subsequence=True)
            if harmful:
                # print(f"{i}: {subsequence}")
                new_batch.append(subsequence)
            else:
                new_batch.append(prompt)
        else:
            new_batch.append(prompt)

    return new_batch

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Print GPU name
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0), flush=True)

def read_text(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
parser.add_argument('--safe_train', type=str, default='data/safe_prompts_train.txt', help='File containing safe prompts for training')
parser.add_argument('--harmful_train', type=str, default='data/harmful_prompts_train.txt', help='File containing harmful prompts for training')
parser.add_argument('--safe_test', type=str, default='data/safe_prompts_test.txt', help='File containing safe prompts for testing')
parser.add_argument('--harmful_test', type=str, default='data/harmful_prompts_test.txt', help='File containing harmful prompts for testing')
parser.add_argument('--save_path', type=str, default='models/distilbert_greedy.pt', help='Path to save the model')

args = parser.parse_args()

# Hyperparameters
batch_size = 32
train_iter = 15
test_iter = 9
num_epochs = 30
num_runs = 10
train_threshold = 0.7

# Read the data
safe_train = read_text(args.safe_train)
harmful_train = read_text(args.harmful_train)
safe_test = read_text(args.safe_test)
harmful_test = read_text(args.harmful_test)

# Create the labels (Class 1: Safe, Class 0: Harmful)
train_texts = safe_train + harmful_train
train_labels = [1]*len(safe_train) + [0]*len(harmful_train)

# Shuffle the data
rand_idx = np.random.permutation(len(train_texts))
train_texts = [train_texts[i] for i in rand_idx]
train_labels = [train_labels[i] for i in rand_idx]

# Validation set
val_size = 0.2 * len(train_texts)
val_texts, val_labels = train_texts[:int(val_size)], train_labels[:int(val_size)]
train_texts, train_labels = train_texts[int(val_size):], train_labels[int(val_size):]

# Test set
test_texts = safe_test + harmful_test
test_labels = [1]*len(safe_test) + [0]*len(harmful_test)

train_len = len(train_texts)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Loss function
loss_fn = nn.CrossEntropyLoss()

best_acc = 0
best_val_loss = float('inf')

for run in range(num_runs):
    print(f"Run {run+1}", flush=True)

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Set the model to the device
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, train_len, batch_size):
            batch_texts = train_texts[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            # Preprocess the batch
            # batch_texts = batch_greedy_ec_test(batch_texts, batch_labels, model, tokenizer, train_iter)
            batch_texts, batch_labels = batch_greedy_ec_train(batch_texts, batch_labels, model, tokenizer, train_iter, train_threshold)
            # print(f"Batch size: {len(batch_texts)}")
            # print(f"Labels: {len(batch_labels)}")

            # Tokenize the batch
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            inputs.to(device)

            # Forward pass
            outputs = model(**inputs, labels=torch.tensor(batch_labels).to(device))
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"    Epoch {epoch+1}, Loss: {total_loss:.2f}", flush=True)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        safe_incorrect = 0
        safe_total = 0

        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i:i+batch_size]
            batch_labels = val_labels[i:i+batch_size]

            # Preprocess the batch
            batch_texts = batch_greedy_ec_test(batch_texts, batch_labels, model, tokenizer, test_iter)
            # batch_texts, batch_labels = batch_greedy_ec_train(batch_texts, batch_labels, model, tokenizer, train_iter, train_threshold)

            # Tokenize the batch
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            inputs.to(device)

            # Forward pass
            outputs = model(**inputs, labels=torch.tensor(batch_labels).to(device))
            loss = outputs.loss

            val_loss += loss.item()

            # Get the predicted labels
            logits = outputs.logits
            predicted_labels = logits.argmax(dim=1)

            # Update the total and correct counts
            total += len(batch_labels)
            safe_total += sum(batch_labels)

            correct += (predicted_labels == torch.tensor(batch_labels).to(device)).sum().item()
            safe_incorrect += sum([1 for i in range(len(batch_labels)) if batch_labels[i] != predicted_labels[i] and batch_labels[i] == 1])

        acc = correct/total
        print(f"    Validation Loss: {val_loss:.2f}, Accuracy: {acc * 100:.2f}%, Safe Incorrect: {safe_incorrect/safe_total * 100:.2f}%", flush=True)

        if val_loss < best_val_loss:
            print("        Saving best model...", flush=True)
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)

    #     if acc > best_acc:
    #         print("        Saving best model...")
    #         # print(f"Best Accuracy: {best_acc * 100:.2f}%")
    #         # print(f"Accuracy: {acc * 100:.2f}%")
    #         best_acc = acc
    #         torch.save(model.state_dict(), args.save_path)

    #     if best_acc == 1:
    #         break

    # if best_acc == 1:
    #     break

# Test the model
model.load_state_dict(torch.load(args.save_path))
model.eval()

test_loss = 0
correct = 0
total = 0
safe_incorrect = 0
safe_total = 0

for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]

    # Preprocess the batch
    batch_texts = batch_greedy_ec_test(batch_texts, batch_labels, model, tokenizer, test_iter)

    # Tokenize the batch
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
    inputs.to(device)

    # Forward pass
    outputs = model(**inputs, labels=torch.tensor(batch_labels).to(device))
    loss = outputs.loss

    test_loss += loss.item()

    # Get the predicted labels
    logits = outputs.logits
    predicted_labels = logits.argmax(dim=1)

    # Update the total and correct counts
    total += len(batch_labels)
    safe_total += sum(batch_labels)

    correct += (predicted_labels == torch.tensor(batch_labels).to(device)).sum().item()
    safe_incorrect += sum([1 for i in range(len(batch_labels)) if batch_labels[i] != predicted_labels[i] and batch_labels[i] == 1])

acc = correct/total
print(f"Test Loss: {test_loss:.2f}, Accuracy: {acc * 100:.2f}%, Safe Incorrect: {safe_incorrect/safe_total * 100:.2f}%", flush=True)
