import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from utils.build_vocab import Vocabulary
import numpy as np
from torchvision import transforms
import pickle
from nltk import word_tokenize
import random
from collections import defaultdict

class ChestXrayDataSet(Dataset):
    def __init__(self,
                 file_list,
                 image_embeddings,
                 groups_path='report2group.pkl',
                 sentence_embeddings_path='sentence_counts.pkl',
                 transforms=None,
                 dataset_name='openi',
                 use_center=False,
                 centers=None,
                 num_negatives=4):
        
        with open(file_list, 'r') as f:
            train_list = json.load(f)
        
        print('load from {}'.format(groups_path))
        with open(groups_path, 'rb') as f:
            self.sid2group, self.group2sid, self.report2sid, self.report2group, self.group2report = pickle.load(f)
        self.num_clusters = len(self.group2sid)
        
        self.total_examples = []
        
        for report_id in train_list:
            sentence_indices = self.report2sid[report_id] 
            for sentence_id in sentence_indices:
                group_id = self.sid2group[sentence_id]
                self.total_examples.append((report_id, sentence_id, group_id))

        # image embedding
        self.image_embeddings = image_embeddings
        
        # sentence embedding
        with open(sentence_embeddings_path, 'rb') as f:
            _, _, self.sent2idx, self.idx2sent, self.sent_embeddings = pickle.load(f)    
        
        self.use_center = use_center
        self.centers = centers
        self.num_negatives = num_negatives
       
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        u, v, group = self.total_examples[index] # image and sentence pair
        u_frontal = self.image_embeddings['{}-0'.format(u)]
        u_lateral = self.image_embeddings['{}-1'.format(u)]
        
        # abnormality groups that report u contains 
        groups = set(self.report2group[u]) 
      
        total_u_negs = []
        total_u_neg_frontal = []
        total_u_neg_lateral = []
        while len(total_u_negs) < self.num_negatives:
            # sample u- which does not include the abnormality group v
            group_u_neg = random.randint(0, self.num_clusters-1)
            # while group_u_neg in groups or len(self.group2report[group_u_neg]) <= 0:
            while group_u_neg == group or len(self.group2report[group_u_neg]) <= 0:
                group_u_neg = random.randint(0, self.num_clusters-1)
            u_neg = random.choice(self.group2report[group_u_neg])
            total_u_negs.append(u_neg)
            total_u_neg_frontal.append(self.image_embeddings['{}-0'.format(u_neg)])
            total_u_neg_lateral.append(self.image_embeddings['{}-1'.format(u_neg)])
        
        total_u_neg_frontal = torch.stack(total_u_neg_frontal, 0)
        total_u_neg_lateral = torch.stack(total_u_neg_lateral, 0)
    
        total_group_v_negs = []
        total_v_negs = []
        while len(total_v_negs) < self.num_negatives:
            # sample v- which is not in the abnormality groups of u
            group_v_neg = random.randint(0, self.num_clusters-1)
            while group_v_neg in groups or len(self.group2sid[group_v_neg]) <= 0:
                group_v_neg = random.randint(0, self.num_clusters-1)
            v_neg = random.choice(list(self.group2sid[group_v_neg]))
            total_group_v_negs.append(group_v_neg)
            total_v_negs.append(v_neg)
         
        total_v_neg_sent = []
        if self.use_center:  
            v_sent = self.centers[group]
            # v_neg_sent = self.centers[group_v_neg]
            for group_v_neg in total_group_v_negs:
                total_v_neg_sent.append(self.centers[group_v_neg])
        else:
            v_sent = torch.tensor(self.sent_embeddings[v]).view(-1)
            for v_neg in total_v_negs:
                # v_neg_sent = torch.tensor(self.sent_embeddings[v_neg])
                total_v_neg_sent.append(torch.tensor(self.sent_embeddings[v_neg]).view(-1))
  
        total_v_neg_sent = torch.stack(total_v_neg_sent, 0)
        
        # return u, v, u_neg, v_neg, u_frontal, u_lateral, v_sent, u_neg_frontal, u_neg_lateral, v_neg_sent
        return u, v, total_u_negs, total_v_negs, u_frontal, u_lateral, v_sent, total_u_neg_frontal, total_u_neg_lateral, total_v_neg_sent

    def __len__(self):
        return len(self.total_examples)


def collate_fn(data):
    u, v, total_u_negs, total_v_negs, u_frontal, u_lateral, v_sent, total_u_neg_frontal, total_u_neg_lateral, total_v_neg_sent = zip(*data)
    
    # B x 1024 x 16 x 16
    u_frontal = torch.stack(u_frontal, 0)
    u_lateral = torch.stack(u_lateral, 0)
    total_u_neg_frontal = torch.stack(total_u_neg_frontal, 0)
    total_u_neg_lateral = torch.stack(total_u_neg_lateral, 0)
    v_sent = torch.stack(v_sent, 0) 
    total_v_neg_sent = torch.stack(total_v_neg_sent, 0)
            
    return u, v, total_u_negs, total_v_negs, u_frontal, u_lateral, v_sent, total_u_neg_frontal, total_u_neg_lateral, total_v_neg_sent


def get_loader(file_list,
               image_embeddings,
               groups_path,
               sentence_embeddings_path,
               transform=None,
               dataset_name='openi',
               batch_size=32,
               shuffle=False,
               n_gpus=1,
               num_workers=1,
               num_negatives=4):
    dataset = ChestXrayDataSet(file_list,
                               image_embeddings,
                               groups_path,
                               sentence_embeddings_path=sentence_embeddings_path,
                               transforms=transform,
                               dataset_name=dataset_name,
                               num_negatives=num_negatives)

    # random sampler
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn,
                                              pin_memory=True)

    return data_loader


if __name__ == '__main__':
    '''python -m utils.rank_pair_dataset'''
    
    data_path = '/home/jianmo/research/medical/report-generation/ranking'
    file_list = os.path.join(data_path, 'data/mimic2/full_train_split.json')
    groups_path = os.path.join(data_path, 'report2group_500.pkl')
    image_embeddings_path = os.path.join(data_path, 'mimic_embeddings.pt')
    sentence_embeddings_path = os.path.join(data_path, 'sentence_counts.pkl')
    
    batch_size = 2
    resize = 256
    crop_size = 224

#     transform = transforms.Compose([
#         transforms.Resize(resize),
#         transforms.RandomCrop(crop_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
                                         

    image_embeddings = {}
    with open(file_list, 'r') as f:
        train_list = json.load(f)
    for report_id in train_list:
        image_embeddings['{}-0'.format(report_id)] = torch.randn((10,1,1))
        image_embeddings['{}-1'.format(report_id)] = torch.randn((10,1,1))
        
    data_loader = get_loader(file_list,
                             image_embeddings,
                             groups_path,
                             sentence_embeddings_path,
                             dataset_name='mimic',
                             batch_size=batch_size,
                             shuffle=False)

    for u, v, total_u_negs, total_v_negs, u_frontal, u_lateral, v_sent, total_u_neg_frontal, total_u_neg_lateral, total_v_neg_sent in data_loader:
        print(u)
        print(v)
        print(total_u_negs)
        print(total_v_negs)
        print(u_frontal.shape)
        print(u_lateral.shape)
        print(v_sent.shape)
        
        print(total_u_neg_frontal.shape)
        print(total_u_neg_lateral.shape)
        print(total_v_neg_sent.shape)
        break
