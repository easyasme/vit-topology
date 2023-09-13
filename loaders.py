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
from config import SUBSETS_LIST, IMG_SIZE

# LARGE_ENOUGH_NUMBER = 1000
# PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def get_transform(train=True, resize=None, crop=None, hflip=True, vflip=True):
    transform = transforms.Compose([])

    if train:
        if not resize == None:
            transform.transforms.insert(0, transforms.Resize(resize, interpolation=Image.BICUBIC))
        if not crop == None:
            transform.transforms.insert(1, transforms.RandomCrop(crop))
        if hflip:
            transform.transforms.insert(2, transforms.RandomHorizontalFlip())
        if vflip:
            transform.transforms.insert(3, transforms.RandomVerticalFlip())
    else:
        if not resize == None:
            transform.transforms.insert(0, transforms.Resize(resize, interpolation=Image.BICUBIC))

    len_transform = len(transform.transforms)
    transform.transforms.insert(len_transform, transforms.ToTensor())

    return transform

def calc_mean_std(dataloader):
    pop_mean = []
    pop_std = []

    for _, data in enumerate(dataloader):
        image_batch = data[0]
        
        batch_mean = image_batch.mean(dim=[0,2,3])
        batch_std = image_batch.std(dim=[0,2,3])
        
        pop_mean.append(batch_mean.numpy())
        pop_std.append(batch_std.numpy())

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)
    
    return pop_mean, pop_std

def get_dataset(data, path, transform, verbose, iter=0):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''
    if data == 'imagenet':
        dataset = CustomImageNet(path, 'data/map_clsloc.txt', subset=SUBSETS_LIST[iter], transform=transform, verbose=verbose)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def dataloader(data, path, transform, batch_size, num_workers, iter, verbose, sampling=-1):
    dataset = get_dataset(data, path, transform, iter=iter, verbose=verbose)
    
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

    print("Transform after: ", transform, "\n")

    return data_loader

def loader(data, batch_size, verbose, iter=0, sampling=-1):
    ''' Interface to the dataloader function '''
    num_workers = os.cpu_count()

    img_size = IMG_SIZE

    if img_size == 32:
        train_data_path = 'data/train_32'
        test_data_path = 'data/val_32'
        transforms_tr_imagenet = get_transform(train=True, hflip=True, vflip=True)
        transforms_te_imagenet = get_transform(train=False, hflip=False, vflip=False)
    elif img_size == 64:
        train_data_path = 'data/train_64'
        test_data_path = 'data/val_64'
        transforms_tr_imagenet = get_transform(train=True, hflip=True, vflip=True)
        transforms_te_imagenet = get_transform(train=False, hflip=False, vflip=False)
    else:
        train_data_path = 'data/train'
        test_data_path = 'data/val'
        transforms_tr_imagenet = get_transform(train=True, resize=(img_size, img_size), hflip=True, vflip=True)
        transforms_te_imagenet = get_transform(train=False, resize=(img_size, img_size), hflip=False, vflip=False)
    
    if data == 'imagenet_train':
        print("\n", "Class Subset: ", SUBSETS_LIST[iter], "(" + str(iter) + ")", "\n")
        return dataloader('imagenet', train_data_path, transform=transforms_tr_imagenet, batch_size=batch_size, num_workers=num_workers, iter=iter, verbose=verbose) 
    elif data == 'imagenet_test':
        return dataloader('imagenet', test_data_path, transform=transforms_te_imagenet, batch_size=batch_size, num_workers=num_workers, iter=iter, verbose=verbose)


class CustomImageNet(Dataset):
    def __init__(self, data_path, labels_path, verbose, subset=[], transform=None, grayscale=False):
        self.data_path = data_path
        self.data = []
        self.label_dict = {}
        self.name_dict = {}
        self.transform = transform
        self.verbose = verbose
        
        if data_path == 'data/train' or data_path == 'data/val':
            img_format = '*.JPEG'
        else:
            img_format = '*.png'

        with open(labels_path, 'r') as f:
            for line in f:
                key = line.split()[0]
                value = int(line.split()[1])
                name = line.split()[2]
                
                if value in subset:
                    self.name_dict[key] = name
                    self.label_dict[key] = value

        for i, key in enumerate(self.label_dict.keys()):
            img_paths = glob.glob(os.path.join(data_path, key, img_format))

            if i in range(9) and self.verbose:
                print("Label mapping:", key + ' --> ' + str(i), " " + self.name_dict[key])
            elif self.verbose:
                print("Label mapping:", key + ' --> ' + str(i), " " + self.name_dict[key], "\n")

            for img_path in img_paths:
                img = Image.open(img_path)
                img = self.transform(img)

                if img.size()[0] == 3 and not grayscale:
                    self.data.append((img, i))
                elif img.size()[0] == 1 and grayscale:
                    self.data.append((img, i))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]