import torchvision.transforms as transforms
import torch
import torchvision
import random
import numpy as np
import os
import glob

from torch.utils.data import *
from PIL import Image
from PIL import PngImagePlugin
from config import SUBSETS_LIST

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

################# Transformers ############################

TRANSFORMS_TR_IMAGENET = transforms.Compose([
    # transforms.RandomCrop(32,  , padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

TRANSFORMS_TE_IMAGENET = transforms.Compose([
    transforms.ToTensor()
])


# def get_transform(train=True, resize=None, crop=None, flip=True):
    # transform = transforms.Compose([])

    # if train:
        # if not resize:
        #     transform.transforms.insert(0, transforms.Resize(resize))
        # if not crop:
        #     transform.transforms.insert(1, transforms.RandomCrop(crop))
        # if flip:
        #     transform.transforms.insert(2, transforms.RandomHorizontalFlip())
    # else:
        # if not resize:
        #     transform.transforms.insert(0, transforms.Resize(resize))

    # len_transform = len(transform.transforms)
    # transform.transforms.insert(len_transform, transforms.ToTensor())

    # return transform

##############################################################

def calc_mean_std(dataloader):
    pop_mean = []
    pop_std = []
    print("Data loader: ", dataloader)
    print("Data loader size: ", len(dataloader))
    for _, data in enumerate(dataloader):
        print("Data shape: ", data[0].size())
        image_batch = data[0]
        
        batch_mean = image_batch.mean(dim=[0,2,3])
        batch_std = image_batch.std(dim=[0,2,3])
        
        pop_mean.append(batch_mean.numpy())
        pop_std.append(batch_std.numpy())

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)
    
    return pop_mean, pop_std

def get_dataset(data, path, train, transform, iter=0):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''
    if data == 'imagenet':
        dataset = CustomImageNet(path, 'data/map_clsloc.txt', subset=SUBSETS_LIST[iter], transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def dataloader(data, path, train, transform, batch_size, num_workers, iter, sampling=-1):
    dataset = get_dataset(data, path, train, transform, iter=iter)
    
    # print("Transform before: ", transform)
        
    if sampling == -1:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)

    if not isinstance(transform.transforms[-1], torchvision.transforms.transforms.Normalize):
        mean, std = calc_mean_std(data_loader)
        # print("Data Mean: ", mean)
        # print("Data SD: ", std, "\n")
        
        len_transform = len(transform.transforms)
        transform.transforms.insert(len_transform, transforms.Normalize(mean, std))

    # print("Transform after: ", transform, "\n")

    return data_loader

def loader(data, batch_size, iter=0, sampling=-1):
    ''' Interface to the dataloader function '''

    num_workers = os.cpu_count()
    
    if data == 'imagenet_train':
        return dataloader('imagenet', './data/train_32', train=True, transform=TRANSFORMS_TR_IMAGENET, batch_size=batch_size, num_workers=num_workers, iter=iter, sampling=sampling)
    elif data == 'imagenet_test':
        return dataloader('imagenet', './data/val_32', train=False, transform=TRANSFORMS_TE_IMAGENET, batch_size=batch_size, num_workers=num_workers, iter=iter, sampling=sampling)

class CustomImageNet(Dataset):
    def __init__(self, data_path, labels_path, subset=[], transform=None):
        self.data_path = data_path
        # print("Data path: ", self.data_path)
        self.data = []
        self.label_dict = {}
        self.transform = transform

        # print("Labels path: ", labels_path)
        with open(labels_path, 'r') as f:
            for line in f:
                key = line.split()[0]
                value = int(line.split()[1])

                if value in subset:
                    self.label_dict[key] = value

        for key in self.label_dict.keys():
            label = self.label_dict[key]
            img_paths = glob.glob(os.path.join(self.data_path, key, '*.png'))
            
            for img_path in img_paths:
                img = Image.open(img_path)
                img = self.transform(img)
                self.data.append((img, label))


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]