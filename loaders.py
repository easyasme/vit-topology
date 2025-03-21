import glob
import os
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms

from PIL import Image 
from PIL import PngImagePlugin
from torch import Generator, initial_seed
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
from config import SEED, IMG_SIZE

# uncomment these lines to allow large images and truncated images to be loaded
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def get_color_distortion(s=0.125): # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return color_distort

def get_transform(train=True, crop=True, hflip=True, vflip=False, color_dis=True, resize=None):
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

def get_dataset(data, path, transform, train=False):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''
    
    if data == 'imagenet':
        dataset = CustomImageNet(path, transform=transform)
    elif data == 'mnist':
        dataset = CustomMNIST(train=train, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def dataloader(data, path=None, train=False, transform=None, batch_size=1, sampling=-1, subset=None):
    
    dataset = get_dataset(data, path, transform, train=train)

    if subset is not None:
        subset_iter = list(np.random.choice(dataset.__len__(), size=subset, replace=True)) # True for bootstrapping
        dataset = Subset(dataset, subset_iter)

    if sampling == -1:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    def seed_worker(worker_id):
        worker_seed = initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = Generator()
    g.manual_seed(SEED)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=4,
                             worker_init_fn=seed_worker,
                             generator=g,
                             drop_last=False)

    return data_loader

def loader(data, batch_size, sampling=-1, subset=None, transform=None):
    ''' Interface to the dataloader function '''

    # set data paths for different image sizes (32, 64, 256)
    train_data_path = '../imagenet_data/train'
    val_data_path = '../imagenet_data/val'
    test_data_path = '../imagenet_data/test'
    
    train = True if 'train' in data else False

    if transform is None:
        transform = get_transform(train=train, crop=True, hflip=True, vflip=False, color_dis=True, blur=True, resize=None)
    
    # return dataloader for different datasets and train/test splits
    if data == 'imagenet_train':
        return dataloader('imagenet', path=train_data_path, train=train, transform=transform, batch_size=batch_size, subset=subset)
    elif data == 'imagenet_val':
        return dataloader('imagenet', path=val_data_path, train=train, transform=transform, batch_size=batch_size, subset=subset)
    elif data == 'imagenet_test':
        return dataloader('imagenet', path=test_data_path, transform=transform, batch_size=batch_size, subset=subset)
    elif data == 'mnist_train':
        return dataloader('mnist', train=train, transform=transform, batch_size=batch_size, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', train=train, transform=transform, batch_size=batch_size, subset=subset)
    else:
        raise ValueError(f"Invalid dataset: {data}")


class CustomImageNet(Dataset):

    def __init__(self, data_path, transform=None):
        super(CustomImageNet, self).__init__()
        
        self.data_path = data_path
        self.image_files = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = torchvision.io.read_image(os.path.join(self.data_path, self.image_files[idx]))

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img   

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
