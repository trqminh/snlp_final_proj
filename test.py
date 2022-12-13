from bbc_datasets import train_bbc_news_dataset, test_bbc_news_dataset
from models import CustomDistilBertForSequenceClassification, CustomBertForSequenceClassification

import argparse
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn import preprocessing
import csv
import os
import sys
import logging
import copy
import time
import wandb
logger = logging.getLogger()

def parse_args(in_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["distilbert", "bert"],
        required=True,
        help="use bert or distilbert",
    )
    parser.add_argument(
        "--dataset",
        choices=["bbc"],
        required=True,
        help="specify datasets",
    )
    
    parser.add_argument('--model_path', type=str, required=True)

    return parser.parse_args(in_args)


def accuracy(output, labels):
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(labels)
    return torch.mean(correct.float())


if __name__=='__main__':
    
    args = parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # dataset
    if args.dataset == 'bbc':
        label_names = ['entertainment', 'business', 'sport', 'politics', 'tech']
        num_classes = len(label_names)
        test_datsaet = test_bbc_news_dataset
    
    batch_size=64
    test_loader = DataLoader(dataset=test_bbc_news_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # model
    if args.model == 'distilbert':
        model = CustomDistilBertForSequenceClassification(num_classes=num_classes)
    else:
        model = CustomBertForSequenceClassification(num_classes=num_classes)

    model = model.to(device)
    print('Model param: ', model.count_params())
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    # test
    test_acc = 0
    since = time.time()
    for samples, labels in test_loader:
        with torch.no_grad():
            samples, labels = samples.to(device), labels.to(device)
            output = model(samples)[0]
            test_acc += accuracy(output, labels)

    print('Accuracy on test set: {}%'.format(round(test_acc.item()*100.0/len(test_loader), 2)))
    
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
