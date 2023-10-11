import torchvision.transforms as transforms
import torch
import torchvision
import random
import numpy as np
import os
import glob
import numpy as np

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from PIL import Image
from PIL import PngImagePlugin
from config import SUBSETS_LIST, IMG_SIZE


# uncomment these lines to allow large images to be loaded
LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# file where mappings from class codes to class names are stored
CODES_TO_NAMES_FILE = './results/codes_to_names.txt'

def get_color_distortion(s=0.125): # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    return color_distort

def get_transform(train=True, crop=True, hflip=True, vflip=False, color_dis=True, blur=True):
    transform = transforms.Compose([])

    if train:
        if crop:
            transform.transforms.insert(0, transforms.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), \
                                                                        interpolation=Image.BICUBIC))
        if hflip:
            transform.transforms.insert(1, transforms.RandomHorizontalFlip())
        if color_dis:
            transform.transforms.insert(2, get_color_distortion())
        # if blur:
        #     transform.transforms.insert(3, transforms.GaussianBlur(kernel_size=IMG_SIZE//20*2+1, sigma=(0.1, 2.0)))
        if vflip:
            transform.transforms.insert(3, transforms.RandomVerticalFlip())

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
        dataset = CustomImageNet(path, 'data/map_clsloc.txt', subset=SUBSETS_LIST[iter], transform=transform, verbose=verbose, iter=iter)
    elif data.split('_')[0] == 'dummy':
        dataset = DummyDataset(transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dataloader(data, path=None, transform=None, batch_size=1, iter=0, verbose=False, sampling=-1):
    dataset = get_dataset(data, path, transform, iter=iter, verbose=verbose)

    if data.split('_')[0] == 'dummy':
        train, test = random_split(dataset, [.7, .3])

        if data.split('_')[1] == 'train':
            dataset = train
        else:
            dataset = test
    
    print("Transform before: ", dataset.__gettransform__(), "\n")

    if sampling == -1:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1, drop_last=True, worker_init_fn=seed_worker)

    if not isinstance(transform.transforms[-1], torchvision.transforms.transforms.Normalize):
        mean, std = calc_mean_std(data_loader)
        # print("Data Mean: ", mean)
        # print("Data SD: ", std, "\n")
        
        len_transform = len(transform.transforms)
        transform.transforms.insert(len_transform, transforms.Normalize(mean, std))

    print("Transform after: ", dataset.__gettransform__(), "\n")

    return data_loader

def loader(data, batch_size, verbose, iter=0, sampling=-1):
    ''' Interface to the dataloader function '''

    if IMG_SIZE == 32:
        train_data_path = '/home/trogdent/imagenet_data/train_32'
        test_data_path = '/home/trogdent/imagenet_data/val_32'
        transforms_tr_imagenet = get_transform(train=True, crop=True, hflip=True, vflip=False, blur=True)
        transforms_te_imagenet = get_transform(train=False, crop=False, hflip=False, vflip=False, blur=False)
    elif IMG_SIZE == 64:
        train_data_path = '/home/trogdent/imagenet_data/train_64'
        test_data_path = '/home/trogdent/imagenet_data/val_64'
        transforms_tr_imagenet = get_transform(train=True, crop=True, hflip=True, vflip=False, blur=True)
        transforms_te_imagenet = get_transform(train=False, crop=False, hflip=False, vflip=False, blur=False)
    else:
        train_data_path = '/home/trogdent/imagenet_data/train'
        test_data_path = '/home/trogdent/imagenet_data/val'
        transforms_tr_imagenet = get_transform(train=True, crop=True, hflip=True, vflip=False, blur=True)
        transforms_te_imagenet = get_transform(train=False, crop=False, hflip=False, vflip=False, blur=False)
    
    if data == 'imagenet_train':
        return dataloader('imagenet', train_data_path, transform=transforms_tr_imagenet, batch_size=batch_size, iter=iter, verbose=verbose) 
    elif data == 'imagenet_test':
        return dataloader('imagenet', test_data_path, transform=transforms_te_imagenet, batch_size=batch_size, iter=iter, verbose=verbose)
    else:
        dummy_transform = get_transform(train=False)

        if data == 'dummy_train':
            return dataloader('dummy_train', transform=dummy_transform, batch_size=batch_size)
        elif data == 'dummy_test':
            return dataloader('dummy_test', tranform=dummy_transform, batch_size=batch_size)


class CustomImageNet(Dataset):

    def __init__(self, data_path, labels_path, verbose, subset=[], transform=None, grayscale=False, iter=0):
        super(CustomImageNet, self).__init__()
        
        self.data_path = data_path
        self.data = []
        self.label_dict = {}
        self.name_dict = {}
        self.transform = transform
        self.verbose = verbose
        
        if data_path == '../../../imagenet_data/train' or data_path == '../../../imagenet_data/val':
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

            lines = []
            if i in range(9) and self.verbose:
                lines.append("Label mapping: " + key + ' --> ' + str(i) + " " + self.name_dict[key] + "\n")
            elif self.verbose:
                lines.append("Label mapping: " + key + ' --> ' + str(i) + " " + self.name_dict[key] + "\n")
                lines.append("End of label mapping for subset " + str(subset) + " " + str(iter) + "\n")
                lines.append("\n")

            with open(CODES_TO_NAMES_FILE, 'a') as f:
                f.writelines(lines)

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

    def __gettransform__(self):
        return self.transform    

class DummyDataset(Dataset):
    
    def __init__(self, transform=None, num_samples=10000):
        super(DummyDataset, self).__init__()

        self.data = []
        self.transform = transform
        self.num_samples = num_samples
    
        for _ in range(num_samples):
            img = np.random.rand(3, 3)
            img, label = self.get_label(img)
            
            self.data.append((img, label))
    
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        img, label = self.data[idx]

        img = self.transform(img)
        label = self.transform(label) # this is a hack to make the label a tensor; the transform is simply ToTensor()
        
        return img, label

    def __gettransform__(self):
        return self.transform

    def get_label(self, img):
        rand_int = np.random.randint(low=0, high=3)
        
        img[rand_int, rand_int] = 0.
        
        label = np.zeros(shape=3, dtype=int)
        label[rand_int] = 1

        return img, label