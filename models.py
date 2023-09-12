import re
import warnings
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast



class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(25, 8)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(8, 2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id):
        x = self.fc1(sent_id.float())

        x = self.relu(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x
