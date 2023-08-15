import torchvision.transforms as transforms
import torch
import torchvision
import random
import numpy as np
import os
import glob

from torch.utils.data import *
from PIL import Image

################# Transformers ############################

# train transform for grayscale images
TRANSFORMS_TR_GRAY = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

TRANSFORMS_TE_GRAY = transforms.Compose([
    transforms.ToTensor()
])

TRANSFORMS_TR_COLOR = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

TRANSFORMS_TR_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

TRANSFORMS_TE_CIFAR10 = transforms.Compose([
    transforms.ToTensor()
])

TRANSFORMS_TR_SVHN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

TRANSFORMS_TE_SVHN = transforms.Compose([
    transforms.ToTensor()
])

TRANSFORMS_TR_IMAGENET = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

TRANSFORMS_TE_IMAGENET = transforms.Compose([
    transforms.ToTensor()
])

TRANSFORMS_MNIST = transforms.Compose([
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
    
    for _, data in enumerate(dataloader):
        image_batch = data[0]
        
        batch_mean = image_batch.mean(dim=[0,2,3])
        batch_std = image_batch.std(dim=[0,2,3])
        
        pop_mean.append(batch_mean.numpy())
        pop_std.append(batch_std.numpy())

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)
    
    return pop_mean, pop_std

def get_dataset(data, path, train, transform):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''

    if data == 'mnist':
        dataset = torchvision.datasets.MNIST(path, train=train, download=True, transform=transform)
    elif data == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transform)
    elif data == 'svhn':
        dataset = torchvision.datasets.SVHN(path, split=train, download=True, transform=transform)
    elif data == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(path, train=train, download=True, transform=transform)
    elif data == 'imagenet':
        dataset = torchvision.datasets.ImageNet(path, split='train' if train else 'val', transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def dataloader(data, path, train, transform, batch_size, num_workers, subset=[], sampling=-1):
    dataset = get_dataset(data, path, train, transform)
    
    # print("Transform before: ", transform)
        
    if sampling == -1:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)

    if not isinstance(transform.transforms[-1], torchvision.transforms.transforms.Normalize):
        mean, std = calc_mean_std(data_loader)
        # print("Data Mean: ", mean)
        # print("Data SD: ", std, "\n")
        
        len_transform = len(transform.transforms)
        transform.transforms.insert(len_transform, transforms.Normalize(mean, std))

    # print("Transform after: ", transform, "\n")

    return data_loader

def loader(data, batch_size, subset=[], sampling=-1):
    ''' Interface to the dataloader function '''

    num_workers = os.cpu_count()

    if data == 'mnist_train':
        return dataloader('mnist', './data', train=True, transform=TRANSFORMS_MNIST, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', './data', train=False, transform=TRANSFORMS_MNIST, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    
    elif data == 'cifar10_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    elif data == 'cifar10_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10, batch_size=batch_size, sampling=sampling , num_workers=num_workers, subset=subset)
    
    elif data == 'svhn_train':
        return dataloader('svhn', './data', train='train', transform=TRANSFORMS_TR_SVHN, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    elif data == 'svhn_test':
        return dataloader('svhn', './data', train='test', transform=TRANSFORMS_TE_SVHN, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    
    elif data == 'fashion_mnist_train':
        return dataloader('fashion_mnist', './data', train=True, transform=TRANSFORMS_TR_GRAY, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    elif data == 'fashion_mnist_test':
        return dataloader('fashion_mnist', './data', train=False, transform=TRANSFORMS_TE_GRAY, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    
    elif data == 'imagenet_train':
        return dataloader('imagenet', './data/train', train=True, transform=TRANSFORMS_TR_IMAGENET, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)
    elif data == 'imagenet_test':
        return dataloader('imagenet', './data/val', train=False, transform=TRANSFORMS_TE_IMAGENET, batch_size=batch_size, sampling=sampling, num_workers=num_workers, subset=subset)

class CustomImageNet(Dataset):
    def __init__(self, data_path, labels_path, subset=[], transform=None):
        self.data_path = data_path
        self.data = []
        self.label_dict = {}
        self.transform = transform

        with open(labels_path, 'r') as f:
            for line in f:
                key = line.split()[0]
                value = int(line.split()[1])

                if value in subset:
                    self.label_dict[key] = value

        for key in self.label_dict.keys():
            imgs = glob.glob(os.path.join(self.data_path, key, '*.png'))
            self.data.append(imgs)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, self.data[idx]))
        img = self.transform(img)

        label = 
        
        return img, 0