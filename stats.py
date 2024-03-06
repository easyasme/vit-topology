import os
import pickle

import matplotlib.pyplot as plt
from PIL import Image

from config import SUBSETS_LIST
from loaders import *


NET = 'lenet'
DATASET = 'imagenet'
SUBSETS = 1 # 1 for none, 30 for all

START = 0 if DATASET.__eq__('mnist') else 0

# img_path_32 = './data/train_32'
# img_path_64 = './data/train_64'

# labels_path = './data/map_clsloc.txt'

# batch_size = 32

# stats = []
# for i, subset in enumerate(SUBSETS_LIST):
#     print("Subset: ", subset)
    
#     transform = get_transform(train=True, crop=False, hflip=False, vflip=False, color_dis=False, blur=False, resize=False)
#     dataset = CustomImageNet(img_path_64, labels_path, verbose=True, subset=subset, transform=transform, grayscale=False, iter=i)
    
#     sampler = RandomSampler(dataset)
#     data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=True,  worker_init_fn=seed_worker)

#     stats.append(calc_mean_std(data_loader)) # pixel-wise mean and std

# print("Means: ", stats[:, 0])
# print("Stds: ", stats[:, 1])

############################################################################################################
''' Make plots of losses and accuracies '''

for iter in range(START, SUBSETS): 
    pkl_path = f"./losses/{NET}/{NET}_{DATASET}_ss{iter}/stats.pkl"

    if not os.path.exists(pkl_path):
        print("No stats.pkl found at", pkl_path)
        continue

    save_dir = f"./results/my_pca/{NET}_{DATASET}_ss{iter}/images/loss/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    acc_save_file = f"{save_dir}/acc.png"
    loss_save_file = f"{save_dir}/loss.png"

    with open(pkl_path, 'rb') as f:
        losses = pickle.load(f)
        
    X = [loss['epoch'] for loss in losses]

    '''Create plots of accuracies'''
    plt.xlabel('Epoch (N)')
    plt.ylabel('Accuracy')
    test_acc = [loss['acc_te']/100. for loss in losses]
    train_acc = [loss['acc_tr']/100. for loss in losses]
    plt.plot(X, test_acc, label='Test')
    plt.plot(X, train_acc, label='Train')
    plt.legend()
    plt.title('Accuracy v. Epoch')
    plt.savefig(acc_save_file)
    plt.clf()

    '''Create plots of losses'''
    plt.xlabel('Epoch (N)')
    plt.ylabel('Loss')
    test_loss = np.array([np.mean(loss['loss_te']) for loss in losses])
    test_std = np.array([np.std(loss['loss_te']) for loss in losses])
    argmin_test_loss = np.argmin(test_loss)
    min_test_loss = test_loss[argmin_test_loss]
    plt.fill_between(X, test_loss - test_std, test_loss + test_std, alpha=0.1, interpolate=True)

    train_loss = np.array([np.mean(loss['loss_tr']) for loss in losses])
    train_std = np.array([np.std(loss['loss_tr']) for loss in losses])
    plt.fill_between(X, train_loss - train_std, train_loss + train_std, alpha=0.1, interpolate=True)
    plt.vlines(X[argmin_test_loss], 0, min_test_loss, linestyles='dashed', label='Min Test Loss')

    plt.plot(X, test_loss, label='Test Mean')
    plt.plot(X, train_loss, label='Train Mean')
    plt.legend()
    plt.title('Average Loss v. Epoch')
    plt.savefig(loss_save_file)
    plt.clf()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    
    return dst

def get_concat_h_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h(_im, im)
    
    return _im

for iter in range(START, SUBSETS):
    save_dir = f"./results/my_pca/{NET}_{DATASET}_ss{iter}/images/curves/"
    files = os.listdir(save_dir)
    files = [f for f in files if f.startswith('epoch')]
    
    print("Directory:", save_dir)

    dims = np.array([int(files[i].split('_')[3]) for i in range(len(files))])
    dims = np.unique(dims)

    if len(dims) == 0:
        print("Empty directory!")
        continue
    elif len(dims) != 1:
        raise ValueError("Files in directory were not calculated with the same upper dimension!")

    for dim in range(dims[0]+1):
        print("Dimension:", dim)
        names = [f for f in files if f.split('_')[-1].startswith(str(dim))] # filter by dimension
        names.sort(key=lambda x: int(x.split('_')[1])) # sort by epoch
        
        images = [Image.open(f"{save_dir}/{f}") for f in names] # open images
        get_concat_h_multi_blank(images).save(f"{save_dir}/concat_dim_{dim}.png") # concat and save
        
        names.clear()
        images.clear()
