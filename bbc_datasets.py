from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer
import numpy as np
from sklearn import preprocessing
import csv
import os
import sys
import logging
import copy
import time
import torch

train_lines = []
with open('data/bbc_news/train.tsv', "r", encoding="utf-8") as f:
  reader = csv.reader(f, delimiter="\t")
  
  for line in reader:
    if sys.version_info[0] == 2:
      line = list(unicode(cell, 'utf-8') for cell in line)
    train_lines.append(line)
  
train_lines = np.array(train_lines)


test_lines = []
with open('data/bbc_news/test.tsv', "r", encoding="utf-8") as f:
  reader = csv.reader(f, delimiter="\t")
  
  for line in reader:
    if sys.version_info[0] == 2:
      line = list(unicode(cell, 'utf-8') for cell in line)
    test_lines.append(line)
  
test_lines = np.array(test_lines)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_length = 128
label_names = ['entertainment', 'business', 'sport', 'politics', 'tech']
le = preprocessing.LabelEncoder()
le.fit(label_names)



class BBCNewsDataset(Dataset):
    def __init__(self, text_inputs, labels):
        self.all_input_ids = list()
        self.text_inputs = text_inputs
        for text in self.text_inputs:
          tokens = tokenizer.tokenize(text)

          if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

          tokens = ["[CLS]"] + tokens + ["[SEP]"]
          input_ids = tokenizer.convert_tokens_to_ids(tokens)
          padding = [0] * (max_seq_length - len(input_ids))
          input_ids += padding

          self.all_input_ids.append(input_ids)

        self.labels = le.transform(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.all_input_ids[index]
        y = self.labels[index]

        return torch.tensor(x), torch.tensor(y)


train_bbc_news_dataset = BBCNewsDataset(train_lines[:, 0], train_lines[:, 1])
test_bbc_news_dataset = BBCNewsDataset(test_lines[:, 0], test_lines[:, 1])
