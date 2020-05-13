import torch
import numpy as np
import csv
import pandas as pd
import transformers
from pathlib import Path
from src.data.datasets import BaseDataset



def sent_tokenize(sentences, tokenizer, MAX_LEN=128):
    ids, attention_masks = [],[]
    for sent in sentences:
        encoded_sent = tokenizer.encode_plus(str(sent), add_special_tokens=True, max_length=MAX_LEN, 
                                             pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])
    ids = torch.cat(ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return ids, attention_masks


class Task1Dataset(BaseDataset):
    """The dataset 
    """

    def __init__(self, data_dir, tokenizer, pretrain_weight, max_len, **kwargs):
        super().__init__(**kwargs)

        tokenizer_class = getattr(transformers, tokenizer)
        self.tokenizer = tokenizer_class.from_pretrained(pretrain_weight, do_lower_case=True)

        if self.type == 'train':
            train = pd.read_csv(str(data_dir / Path('train.tsv')),delimiter='\t',header=None,names=['ids','label','alpha','sentence'])
            self.sents, self.labels = train.sentence.values, train.label.values
            self.inputs, self.masks = sent_tokenize(self.sents, self.tokenizer, max_len)
            self.labels = torch.tensor(self.labels).to(torch.int64)
        elif self.type == 'valid':
            val = pd.read_csv(str(data_dir / Path('val.tsv')),delimiter='\t',header=None,names=['ids','label','alpha','sentence'])
            self.sents, self.labels = val.sentence.values, val.label.values
            self.inputs, self.masks = sent_tokenize(self.sents, self.tokenizer, max_len)
            self.labels = torch.tensor(self.labels).to(torch.int64)
        else:
            raise Exception('The type of dataset is undefined!')

    def __getitem__(self, index):
        inp = self.inputs[index]
        gt = self.labels[index]
        mask = self.masks[index]
        
        metadata = {'inputs': inp, 'targets': gt, 'masks': mask}

        return metadata

    def __len__(self):
        return self.labels.size(0)
