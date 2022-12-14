import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import time
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from collections import Counter
from sklearn.metrics import accuracy_score

import gzip
import gensim
import os
import sys
import json
import shutil
import re
import tarfile
import zipfile


import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize




def review_preprocess(review):
    """
    Takes in a string of review, then performs the following:
    1. Remove HTML tag from review
    2. Remove URLs from review
    3. Make entire review lowercase
    4. Split the review in words
    5. Remove all punctuation
    6. Remove empty strings from review
    7. Remove all stopwords
    8. Returns a list of the cleaned review after jioning them back to a sentence
    """
    en_stops = set(stopwords.words('english'))
    
    """
    Removing HTML tag from review
    """
    clean = re.compile('<.*?>')
    review_without_tag = re.sub(clean, '', review) 
    
    
    """
    Removing URLs
    """
    review_without_tag_and_url = re.sub(r"http\S+", "", review_without_tag)
    
    review_without_tag_and_url = re.sub(r"www\S+", "", review_without_tag)
    
    """
    Make entire string lowercase
    """
    review_lowercase = review_without_tag_and_url.lower()
    
    """
    Split string into words
    """
    list_of_words = word_tokenize(review_lowercase)
    
    
    """
    Remove punctuation
    Checking characters to see if they are in punctuation
    """

    list_of_words_without_punctuation=[''.join(this_char for this_char in this_string if (this_char in string.ascii_lowercase))for this_string in list_of_words]
     
    
    """
    Remove empty strings
    """
    list_of_words_without_punctuation = list(filter(None, list_of_words_without_punctuation))
    
    
    """
    Remove any stopwords
    """
  
    filtered_word_list = [w for w in list_of_words_without_punctuation if w not in en_stops] 
    
    """
    Returns a list of the cleaned review after jioning them back to a sentence
    """
    return ' '.join(filtered_word_list)


"""
Load file into memory
"""
def load_file(filename):
    """
    Open the file as read only
    """
    file = open(filename, 'r')
    """
    Read all text
    """
    text = file.read()
    """
    Close the file
    """
    file.close()
    return text

def get_data(directory, vocab, is_trian):
    """
    Reading train test directory
    """
    review_dict={'neg':[],'pos':[]}
    if is_trian:
        directory = os.path.join(directory+'/train')
    else:
        directory = os.path.join(directory+'/test')
    print('Directory : ',directory)
    for label_type in ['neg', 'pos']: 
            data_folder=os.path.join(directory, label_type)
            print('Data Folder : ',data_folder)
            for root, dirs, files in os.walk(data_folder):
                for fname in files:
                    if fname.endswith(".txt"):
                        file_name_with_full_path=os.path.join(root, fname)
                        review=load_file(file_name_with_full_path)
                        clean_review=review_preprocess(review)
                        if label_type == 'neg':
                            review_dict['neg'].append(clean_review)
                        else:
                            review_dict['pos'].append(clean_review)
                        """
                        Update counts
                        """
                        vocab.update(clean_review.split())
                        
    return review_dict


"""
Define vocab
"""
startTime = time.time()
vocab = Counter()
directory='./data/aclImdb_v1/aclImdb'
train_review_dict=get_data(directory, vocab, True)
test_review_dict=get_data(directory, vocab, False)
total_time=time.time()-startTime
print('Time Taken : ',total_time/60,'minutes')

print('Number of negative reviews in train set :',len(train_review_dict['neg']))
print('Number of positive reviews in train set :',len(train_review_dict['pos']))
print('\nNumber of negative reviews in test set :',len(test_review_dict['neg']))
print('Number of positive reviews in test set :',len(test_review_dict['pos']))

word_list = sorted(vocab, key = vocab.get, reverse = True)
vocab_to_int = {word:idx+1 for idx, word in enumerate(word_list)}
int_to_vocab = {idx:word for word, idx in vocab_to_int.items()}

class IMDBReviewDataset(Dataset):

    def __init__(self, review_dict, alphabet):

        self.data = review_dict
        self.labels = [x for x in review_dict.keys()]
        self.alphabet = alphabet

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        label = 0
        while idx >= len(self.data[self.labels[label]]):
            idx -= len(self.data[self.labels[label]])
            label += 1
        reviewText = self.data[self.labels[label]][idx]



        label_vec = torch.zeros((1), dtype=torch.long)
        label_vec[0] = label
        return self.reviewText2InputVec(reviewText), label

    def reviewText2InputVec(self, review_text):
        T = len(review_text)

        review_text_vec = torch.zeros((T), dtype=torch.long)
        encoded_review=[]
        for pos,word in enumerate(review_text.split()):
            if word not in vocab_to_int.keys():
                """
                If word is not available in vocab_to_int dict puting 0 in that place
                """
                review_text_vec[pos]=0
            else:
                review_text_vec[pos]=vocab_to_int[word]

        return review_text_vec


def pad_and_pack(batch):
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #Assume shape is (T, *)
    longest = max(lengths)
    #We need to pad all the inputs up to 'longest', and combine into a batch ourselves
    if len(input_tensors[0].shape) == 1:
        x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    else:
        raise Exception('Current implementation only supports (T) shaped data')

    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)

    y_batched = torch.as_tensor(labels, dtype=torch.long)

    return torch.einsum('ij->ji',x_padded), y_batched


train_review_dict['pos']=train_review_dict['pos'][:int(len(train_review_dict['pos'])*.5)]
train_review_dict['neg']=train_review_dict['neg'][:int(len(train_review_dict['neg'])*.5)]
print('After Down Sampling the training set :')
print('Number of negative reviews in train set :',len(train_review_dict['neg']))
print('Number of positive reviews in train set :',len(train_review_dict['pos']))


train_imdb_dataset=IMDBReviewDataset(train_review_dict,vocab)
test_imdb_dataset=IMDBReviewDataset(test_review_dict,vocab)
