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


class ChestXrayImageDataSet(Dataset):
    def __init__(self,
                 file_list,
                 image_embeddings,
                 report2group,
                 num_clusters=500,
                 transforms=None,
                 dataset_name='openi'):

        # list of report_id
        with open(file_list, 'r') as f:
            self.report_ids = json.load(f) 
        print("length of report2group = {}".format(len(report2group)))
        
        self.image_embeddings = image_embeddings
        self.report2group = report2group
        self.num_clusters = num_clusters
        
        self.dataset_name = dataset_name

    def __getitem__(self, index, debug=False):
        report_id = self.report_ids[index]
        
        if debug:
            frontal_image_embedding = torch.randn((1024, 16, 16))
            lateral_image_embedding = torch.randn((1024, 16, 16))
        else:
            frontal_image_embedding = self.image_embeddings['{}-0'.format(report_id)]
            lateral_image_embedding = self.image_embeddings['{}-1'.format(report_id)]
        
        label = self.report2group[report_id] # a list of groups
        target = torch.zeros(self.num_clusters, ) # (500,)
        for l in label:
            target[l] = 1
        
        return report_id, frontal_image_embedding, lateral_image_embedding, target

    def __len__(self):
        return len(self.report_ids)


def collate_fn_image(data):
    report_id, frontal_image_embedding, lateral_image_embedding, target = zip(*data)
    
    # B x 1024 x 16 x 16
    images_frontal = torch.stack(frontal_image_embedding, 0)
    images_lateral = torch.stack(lateral_image_embedding, 0)
    
    target = torch.stack(target, 0)         
        
    return report_id, images_frontal, images_lateral, target


def get_loader(file_list,
               image_embeddings,
               report2group,
               num_clusters=500,
               transform=None,
               dataset_name='openi',
               batch_size=32,
               shuffle=False,
               n_gpus=1,
               num_workers=1):
    dataset = ChestXrayImageDataSet(file_list,
                                    image_embeddings,
                                    report2group,
                                    num_clusters=500,
                                    transforms=transform,
                                    dataset_name=dataset_name)

    # random sampler
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn_image,
                                              pin_memory=True)

    return data_loader


if __name__ == '__main__':
    '''python -m utils.rank_image_dataset'''
    
    data_path = '/home/jianmo/research/medical/report-generation/ranking'
    file_list = os.path.join(data_path, 'data/mimic2/full_train_split.json')
    report2group_path = os.path.join(data_path, 'report2group_500.pkl')
    
    num_clusters = 500
    
    batch_size = 4
    resize = 256
    crop_size = 224

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
                  
    with open(report2group_path, 'rb') as f:
            _, _, _, report2group, _ = pickle.load(f)

    report_ids = list(report2group.keys())        
    image_embeddings = {}
        
    data_loader = get_loader(file_list,
                             image_embeddings,
                             report2group,
                             num_clusters,
                             transform=transform,
                             dataset_name='mimic',
                             batch_size=batch_size,
                             shuffle=False)

    for i, (report_id, images_frontal, images_lateral, label) in enumerate(data_loader):
        print(report_id)
        print(images_frontal.shape)
        print(images_lateral.shape)
        print(label.shape)
        break
