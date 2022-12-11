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

    return parser.parse_args(in_args)


def accuracy(output, labels):
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(labels)
    return torch.mean(correct.float())

if __name__=='__main__':
    args = parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    # dataset
    if args.dataset == 'bbc':
        label_names = ['entertainment', 'business', 'sport', 'politics', 'tech']
        num_classes = len(label_names)
        train_dataset = train_bbc_news_dataset
        test_datsaet = test_bbc_news_dataset

    # model
    if args.model == 'distilbert':
        model = CustomDistilBertForSequenceClassification(num_classes=num_classes)
    else:
        model = CustomBertForSequenceClassification(num_classes=num_classes)

    print('Model param: ', model.count_params())

    train_loader = DataLoader(dataset=train_bbc_news_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_bbc_news_dataset, batch_size=64, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000, 15000], gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_acc = 0, 0
    loss_list = []
    acc_list = []

    epochs = 5
    itr = 1
    p_itr = 10

    # start training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    for epoch in range(epochs):
        model.train()
        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(samples)[0]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy(output, labels)
            scheduler.step()

            if itr % p_itr == 0:
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, train_acc: {:.3f}'\
                        .format(epoch + 1, epochs, itr, total_loss / p_itr, total_acc / p_itr))
          
                loss_list.append(total_loss / p_itr)
                acc_list.append(total_acc / p_itr)
                total_loss, total_acc = 0, 0

            itr += 1

        model.eval()
        test_acc = 0.0
        for samples, labels in test_loader:
            with torch.no_grad():
                samples, labels = samples.to(device), labels.to(device)
                output = model(samples)[0]
                test_acc += accuracy(output, labels)

        print('Accuracy on test set after {} epoch: {}%'.format(epoch + 1,
                                      round(test_acc.item()*100.0/len(test_loader), 2)))

        if (test_acc.item() > best_acc):
            best_acc = test_acc.item()
            best_model_wts = copy.deepcopy(model.state_dict())
            print('update best')

        print('-' * 10)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best acc on test set: ', best_acc/len(test_loader))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './{}_{}.pth'.format(args.model, args.dataset)) 
    print('Done!')
