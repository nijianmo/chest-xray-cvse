import os
import torch
import json

import numpy as np

from torch.utils.data import Dataset
from datetime import datetime
from itertools import chain

from transformers import BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features

class AbnormalityClassificationDataset(Dataset):
    """
    Dataset supporting sentence-level splitting in a conversation

    Use case: Apply dialogue history
    """

    def __init__(self, args, split):
        """
        Arguments:
            
        """
        super().__init__()

        self.args = args
        self.split = split
        self.data = []
      
        save_path = args.data_dir
        if 'mimic' in save_path:
            with open(os.path.join(save_path, '{}.json'.format(split)), 'r') as f:
                for l in f:
                    d = json.loads(l.strip())
                    id = d['id']
                    sent = d['sent']
                    label = d['label']
                    self.data.append((id, sent, label))
        else:
            with open(os.path.join(save_path, '{}.json'.format(split)), 'r') as f:
                for l in f:
                    # flatten
                    d = json.loads(l.strip())
                    id = d['id']
                    i = 0
                    for sent, label in zip(d['sents'], d['labels']): 
                        self.data.append(('{}_{}'.format(id, i), sent, label))
                        i += 1
                    
        self.tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        self.max_length = args.max_seq_length
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        self.pad_token_segment_id = 0
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        example = self.data[index]
        
        inputs = self.tokenizer.encode_plus(example[1], add_special_tokens=True, max_length=self.max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_length - len(input_ids)
        
        input_ids = input_ids + ([self.pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([self.pad_token_segment_id] * padding_length)
        
        index = example[0]
        label = example[2]
              
        return input_ids, attention_mask, token_type_ids, label, index
        

def collate_abnormality_classification(batch):
    """
    Arguments:
        batch (list): List of list of tokens

    Returns:
        torch.LongTensor: Padded batch tokens, of shape B x T
        torch.BoolTensor: Padding batch mask, of shape B x T
    """
    input_ids, attention_mask, token_type_ids, label, index = zip(*batch)

    # Pad and mask
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.BoolTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    label = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, label, index



