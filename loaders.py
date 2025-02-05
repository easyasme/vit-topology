import glob
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image 
from PIL import PngImagePlugin
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, Subset, random_split)

from config import SEED

# uncomment these lines to allow large images and truncated images to be loaded
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# file where mappings from class codes to class names are stored
CODES_TO_NAMES_FILE = './results/codes_to_names.txt'

def get_color_distortion(s=0.125): # s is the strength of color distortion.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return color_distort

def get_transform(train=True, crop=True, hflip=True, vflip=False, color_dis=True, blur=True, resize=None):
    transform = transforms.Compose([])

    if train:
        if crop:
            len_trans = len(transform.transforms)
            # transform.transforms.insert(len_trans, transforms.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), \
                                                                        # interpolation=Image.BICUBIC))
            transform.transforms.insert(len_trans, transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)))
        if hflip:
            len_trans = len(transform.transforms)
            transform.transforms.insert(len_trans, transforms.RandomHorizontalFlip())
        if color_dis:
            len_trans = len(transform.transforms)
            transform.transforms.insert(len_trans, get_color_distortion())
        if vflip:
            len_trans = len(transform.transforms)
            transform.transforms.insert(len_trans, transforms.RandomVerticalFlip())
        if resize is not None:
            len_trans = len(transform.transforms)
            transform.transforms.insert(len_trans, transforms.Resize((resize, resize), interpolation=Image.BICUBIC))
    else:
        if resize is not None:
            len_trans = len(transform.transforms)
            transform.transforms.insert(len_trans, transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.BICUBIC))

    len_trans = len(transform.transforms)
    transform.transforms.insert(len_trans, transforms.ToTensor())
    transform.transforms.insert(len_trans+1, transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std=[0.229, 0.224, 0.225]))

    del len_trans

    return transform

def get_dataset(data, path, transform, verbose, train=False):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''
    
    if data == 'imagenet':
        dataset = CustomImageNet(path, './data/cls_map.txt', transform=transform, verbose=verbose)
    elif data == 'mnist':
        dataset = CustomMNIST(train=train, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def dataloader(data, path=None, train=False, transform=None, batch_size=1, verbose=False, sampling=-1, subset=None):
    
    dataset = get_dataset(data, path, transform, train=train, verbose=verbose)

    if subset is not None:
        subset_iter = list(np.random.choice(dataset.__len__(), size=subset, replace=True)) # True for bootstrapping
        dataset = Subset(dataset, subset_iter)

    if sampling == -1:
        print(f'Using RandomSampler, train is {train}')
        sampler = RandomSampler(dataset)
    else:
        print(f'Using SequentialSampler, train is {train}')
        sampler = SequentialSampler(dataset)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, worker_init_fn=seed_worker, generator=g, drop_last=True)

    return data_loader

def loader(data, batch_size, verbose, sampling=-1, subset=None, transform=None):
    ''' Interface to the dataloader function '''

    # set data paths for different image sizes (32, 64, 256)
    train_data_path = '~/groups/grp_dnn_topo/nobackup/archive/imagenet_data/train'
    val_data_path = '~/groups/grp_dnn_topo/nobackup/archive/imagenet_data/val'
    test_data_path = '~/groups/grp_dnn_topo/nobackup/archive/imagenet_data/test'
    
    train = True if 'train' in data else False

    if transform is None:
        transform = get_transform(train=train, crop=True, hflip=True, vflip=False, color_dis=True, blur=True, resize=None)
    
    # return dataloader for different datasets and train/test splits
    if data == 'imagenet_train':
        return dataloader('imagenet', path=train_data_path, train=train, transform=transform, batch_size=batch_size, verbose=verbose, subset=subset)
    elif data == 'imagenet_val':
        return dataloader('imagenet', path=val_data_path, train=train, transform=transform, batch_size=batch_size, verbose=verbose, subset=subset)
    elif data == 'imagenet_test':
        return dataloader('imagenet', path=test_data_path, transform=transform, batch_size=batch_size, verbose=verbose, subset=subset)
    elif data == 'mnist_train':
        return dataloader('mnist', train=train, transform=transform, batch_size=batch_size, verbose=verbose, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', train=train, transform=transform, batch_size=batch_size, verbose=verbose, subset=subset)
    else:
        raise ValueError(f"Invalid dataset: {data}")


class CustomImageNet(Dataset):

    def __init__(self, data_path, verbose, subset=[], transform=None, num_samples=10000):
        super(CustomImageNet, self).__init__()
        
        self.data_path = data_path
        self.image_files = glob.glob(data_path, '*.JPEG')
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.image_files[idx])

        img.load()

        img = self.transform(img) if self.transform else transforms.ToTensor()(img)

        return img

    def __settransform__(self, transform):
        self.transform = transform

    def __gettransform__(self):
        return self.transform    

class CustomMNIST(Dataset):
    def __init__(self, path='./data/mnist', train=True, transform=None):
        super(CustomMNIST, self).__init__()

        download = False if os.path.exists(path) else True
        os.makedirs(path, exist_ok=True)

        self.data = torchvision.datasets.MNIST(path, train=train, download=download, transform=transform)

    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    def __gettransform__(self):
        if hasattr(self.data, 'transform'):
            return getattr(self.data, 'transform')
        else:
            raise AttributeError("CustomMNIST has no attribute 'transform'")