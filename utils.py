
import torch,torchtext,csv
from torch.utils.data import Dataset, DataLoader

def _split(input_str):
    return input_str.split('/')[1:]

def _split_rela(input_str):
    input_str = input_str.split(':')[1]
    input_str = input_str.replace('_','.')
    return input_str.split('.')

def padding(tokenizer, text, length):
    ret = tokenizer.convert_tokens_to_ids(text)
    pad = [0] * (length - len(ret))
    return ret + pad


def get_dataset(data_path = "./dataset_sq/"):
    TEXT = torchtext.data.Field()
    ED = torchtext.data.Field()

    train, dev, test = torchtext.data.TabularDataset.splits(path=data_path, train='train.txt', validation='valid.txt', test='test.txt', format='tsv', fields=[('id', TEXT), ('sub_m', TEXT), ('sub', TEXT),('rela', TEXT),('obj_m', TEXT),('text', TEXT)], csv_reader_params={'quoting':csv.QUOTE_NONE})
    
    for t in train:
        t.rela = _split_rela(t.rela[0])

    for t in dev:
        t.rela = _split_rela(t.rela[0])
    
    for t in test:
        t.rela = _split_rela(t.rela[0])

    return train,dev,test



class BuboDataset(Dataset):
    def __init__(self, dataset):
        self.question = []
        self.choices = []
        self.label = []
        for datum in dataset:
            self.question.append(datum.text)
            self.choices.append(datum.choice)
            self.label.append(datum.label)

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        return self.question[idx], self.choices[idx], self.label[idx]


def prompt(q, sub, rela, method="none"):
    if method == "none" :
        return q + rela
    elif method == "base" :
        # return ["A", "question", "is"] + q + [",", "the", "answer", "is"] + rela
        res = ["A", "question", "is"] + q + [",", "the", "relation", "in", "this" , "question","is"] + rela + [",", "the", "entity", "in", "this" , "question","is"] + sub
        res = res[:50]
        return res
    

def generate_batch(batch):
    question, choices, label = zip(*batch)
    return torch.tensor(question, dtype=torch.long), torch.tensor(choices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

import numpy as np

def calc_acc(preds, labels):

    pred_labels = np.argmax(preds, axis=1)
    correct_predictions = np.sum(pred_labels == labels)
    accuracy = correct_predictions / len(labels)
    return accuracy
