import torch
import torch.nn as nn

from transformers import (
    DistilBertConfig, BertTokenizer, \
    DistilBertForSequenceClassification, DistilBertTokenizer, 
    DistilBertModel, BertForSequenceClassification
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CustomDistilBertForSequenceClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return count_parameters(self.model)


class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return count_parameters(self.model)