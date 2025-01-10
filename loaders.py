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

from config import IMG_SIZE, SUBSETS_LIST, SEED

# uncomment these lines to allow large images and truncated images to be loaded
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# file where mappings from class codes to class names are stored
CODES_TO_NAMES_FILE = './results/codes_to_names.txt'


def get_dataset(data, path, transform, verbose, train=False, it=0):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''
    if data == 'imagenet':
        dataset = CustomImageNet(path, './data/map_cls.txt', subset=SUBSETS_LIST[it], transform=transform, verbose=verbose, it=it)
    elif data == 'mnist':
        dataset = CustomMNIST(train=train, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def dataloader(data, path=None, train=False, transform=None, batch_size=1, it=0, verbose=False, sampling=-1, \
               normalize=True, subset=None):
    
    dataset = get_dataset(data, path, transform, train=train, verbose=verbose, it=it)

    if subset is not None:
        subset_iter = list(np.random.choice(dataset.__len__(), size=subset, replace=False))
        dataset = CustomSubset(dataset, subset_iter)

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

    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1, worker_init_fn=seed_worker, generator=g, drop_last=True)

    return data_loader

def loader(data, batch_size, verbose, it=0, sampling=-1, subset=None, transform=None):
    ''' Interface to the dataloader function '''

    # set data paths for different image sizes (32, 64, 256)
    if IMG_SIZE == 32:
        train_data_path = '/home/trogdent/imagenet_data/train_32'
        test_data_path = '/home/trogdent/imagenet_data/val_32'
    elif IMG_SIZE == 64:
        train_data_path = '/home/trogdent/imagenet_data/train_64'
        test_data_path = '/home/trogdent/imagenet_data/val_64'
    else:
        train_data_path = '/home/trogdent/imagenet_data/train'
        test_data_path = '/home/trogdent/imagenet_data/val'
    
    # return dataloader for different datasets and train/test splits
    if data == 'imagenet_train':
        return dataloader('imagenet', path=train_data_path, train=True, transform=transform, batch_size=batch_size, it=it, verbose=verbose, subset=subset) 
    elif data == 'imagenet_test':
        return dataloader('imagenet', test_data_path, transform=transform, batch_size=batch_size, it=it, verbose=verbose, subset=subset)
    elif data == 'mnist_train':
        return dataloader('mnist', train=True, transform=transform, batch_size=batch_size, it=it, verbose=verbose, normalize=True, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', train=False, transform=transform, batch_size=batch_size, it=it, verbose=verbose, normalize=True, subset=subset)
    else:
        raise ValueError(f"Invalid dataset: {data}")


class CustomImageNet(Dataset):

    def __init__(self, data_path, labels_path, verbose, subset=[], transform=None, grayscale=False, it=0, num_samples=20000):
        super(CustomImageNet, self).__init__()
        
        self.data_path = data_path
        self.data = []
        self.label_dict = {}
        self.name_dict = {}
        self.transform = transform
        self.verbose = verbose

        if data_path == '/home/trogdent/imagenet_data/train' or data_path == '/home/trogdent/imagenet_data/val':
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

        lines = []
        lines.append(f'Subset number: {it}\n')
        for i, key in enumerate(self.label_dict.keys()):
            img_paths = glob.glob(os.path.join(data_path, key, img_format))

            if i in range(9) and self.verbose:
                lines.append(f'Label mapping: {key} --> {i} {self.name_dict[key]}\n')
            elif self.verbose:
                lines.append(f'Label mapping: {key} --> {i} {self.name_dict[key]}\n')
                lines.append(f'Original subset labels: {subset}\n')

            counter = 0
            for img_path in img_paths:
                try:
                    img = Image.open(img_path)
                    img.load()
                except OSError:
                    print("Cannot open: {}".format(img_path))
                    continue

                if counter > num_samples:
                    break

                if img.mode == 'RGB' and not grayscale:
                    self.data.append((img, i))
                elif img.mode == 'L' and grayscale:
                    self.data.append((img, i))

                counter += 1

        if not os.path.exists(CODES_TO_NAMES_FILE):
                with open(CODES_TO_NAMES_FILE, 'w') as f:
                    f.writelines(lines)
        else:
            with open(CODES_TO_NAMES_FILE, 'r') as f:
                existing_lines = f.readlines()
            if lines[-1] not in existing_lines:
                with open(CODES_TO_NAMES_FILE, 'a') as f:
                    f.writelines(lines)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.data[idx][1]

        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        label = torch.tensor(label, dtype=torch.long)

        return (img, label)

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

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)

    def __settransform__(self, transform):
        if hasattr(self.dataset, '__settransform__'):
            self.dataset.__settransform__(transform)
        else:
            raise AttributeError("CustomSubset has no attribute '__settransform__'")

    def __gettransform__(self):
        return self.dataset.__gettransform__()
