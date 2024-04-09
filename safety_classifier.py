# The following code is adapted from:
# 1. HuggingFace tutorial on using DistillBert https://huggingface.co/distilbert/distilbert-base-uncased
# 2. Huggingface tutorial on training transformers for sequence classification here: https://huggingface.co/docs/transformers/tasks/sequence_classification

### Importing libraries
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, SequentialSampler

# specify the available devices
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

## Function for reading the given file
def read_text(filename):	
  with open(filename, "r") as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
  return pd.DataFrame(lines)

# Set seed
seed = 912

## Parser for setting input values
parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
parser.add_argument('--safe_train', type=str, default='data/safe_prompts_train_insertion_erased.txt', help='File containing safe prompts for training')
parser.add_argument('--harmful_train', type=str, default='data/harmful_prompts_train.txt', help='File containing harmful prompts for training')
parser.add_argument('--safe_test', type=str, default='data/safe_prompts_test_insertion_erased.txt', help='File containing safe prompts for testing')
parser.add_argument('--harmful_test', type=str, default='data/harmful_prompts_test.txt', help='File containing harmful prompts for testing')
parser.add_argument('--save_path', type=str, default='models/distilbert_insertion.pt', help='Path to save the model')

args = parser.parse_args()

# Load safe and harmful prompts and create the dataset for training classifier
# Class 1: Safe, Class 0: Harmful
safe_prompt_train = read_text(args.safe_train)
harm_prompt_train = read_text(args.harmful_train)
prompt_data_train = pd.concat([safe_prompt_train, harm_prompt_train], ignore_index=True)
prompt_data_train['Y'] = pd.Series(np.concatenate([np.ones(safe_prompt_train.shape[0]), np.zeros(harm_prompt_train.shape[0])])).astype(int)

# Split train dataset into train and validation sets
train_text, val_text, train_labels, val_labels = train_test_split(prompt_data_train[0], 
								prompt_data_train['Y'], 
								random_state=seed, 
								test_size=0.2,
								stratify=prompt_data_train['Y'])

# Count number of samples in each class in the training set
count = train_labels.value_counts().to_dict()

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# pass the pre-trained DistilBert to our define architecture
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
# print(model)

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

## Convert lists to tensors for train split
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())
sample_weights = torch.tensor([1/count[i] for i in train_labels])

## Convert lists to tensors for validation split
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# define the batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
# train_sampler = RandomSampler(train_data)
train_sampler = WeightedRandomSampler(sample_weights, len(train_data), replacement=True)

# dataLoader for the train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# push the model to GPU
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)          # learning rate

# from sklearn.utils.class_weight import compute_class_weight

# #compute the class weights
# class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels.to_numpy())

# print("Class Weights:",class_weights)

# # converting list of class weights to a tensor
# weights= torch.tensor(class_weights,dtype=torch.float)

# # push to GPU
# weights = weights.to(device)

# define the loss function
# loss_fn  = nn.NLLLoss(weight=weights) 
# loss_fn  = nn.CrossEntropyLoss(weight=weights)
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.NLLLoss()

# number of training epochs
epochs = 10

# function to train the model
def train():

  model.train()
  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step, batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if (step + 1) % 50 == 0 or step == len(train_dataloader) - 1:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step + 1, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()     

    # get model predictions for the current batch
    preds = model(sent_id, mask)[0]

    # compute the loss between actual and predicted values
    loss = loss_fn(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds

# function for evaluating the model
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
    if (step + 1) % 50 == 0 or step == len(val_dataloader) - 1:
      
      # Calculate elapsed time in minutes.
      # elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step + 1, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)[0]

      # compute the validation loss between actual and predicted values
      loss = loss_fn(preds, labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# set initial loss to infinite
best_validation_loss = float('inf')

# empty lists to store training and validation loss of each epoch
training_losses=[]
validation_losses=[]
train_flag = True

if train_flag == True:
    # for each epoch
    for epoch in range(epochs):
        # Copilot Code Reference: Similar code with 2 license types [MIT, unknown]
        # https://github.com/github-copilot/code_referencing?cursor=ca31ec3ebd8e24ea9127b39656a9ec6b&editor=vscode
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
      
        #train model
        training_loss, _ = train()
      
        #evaluate model
        validation_loss, _ = evaluate()
      
        #save the best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), args.save_path)
            # torch.save(model.state_dict(), 'new_distillbert_saved_weights.pt')
        
        # append training and validation loss
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        
        print(f'\nTraining Loss: {training_loss:.3f}')
        print(f'Validation Loss: {validation_loss:.3f}')


# Test safety classifier
safe_prompt_test = read_text(args.safe_test)
harm_prompt_test = read_text(args.harmful_test)
prompt_data_test = pd.concat([safe_prompt_test, harm_prompt_test], ignore_index=True)
prompt_data_test['Y'] = pd.Series(np.concatenate([np.ones(safe_prompt_test.shape[0]), np.zeros(harm_prompt_test.shape[0])])).astype(int)

test_text = prompt_data_test[0]
test_labels = prompt_data_test['Y']

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

#load weights of best model
path = args.save_path
# path = 'new_distillbert_saved_weights.pt'
model.load_state_dict(torch.load(path))
model.eval()

# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))[0]
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(f'Testing Accuracy = {100*torch.sum(torch.tensor(preds) == test_y)/test_y.shape[0]}%')
print(classification_report(test_y, preds))
