import os
import pickle

import matplotlib.pyplot as plt
from PIL import Image

from loaders import *

NET = 'alexnet'
DATASET = 'imagenet'
START = 0
SUBSETS = 30 # 1 for none, 30 for all

SAVE_DIR = './results'
RED = 'kmeans' # 'pca' or 'umap' or None
METRIC = 'spearman' # 'euclidean' or 'cosine' or None

SAVE_DIR += f'/{RED}' if RED is not None else ''
SAVE_DIR += f'/{METRIC}' if METRIC is not None else ''

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    
    return dst

def get_concat_v_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_v(_im, im)
    
    return _im

# ''' Make plots of losses and accuracies '''
# for iter in range(START, SUBSETS):
#     print(f'Processing losses for subset {iter}')
#     if DATASET.__eq__('imagenet'):
#         pkl_path = f"./losses/{NET}/{NET}_{DATASET}_ss{iter}/stats.pkl"
#     else:
#         pkl_path = f"./losses/{NET}/{NET}_{DATASET}/stats.pkl"

#     if not os.path.exists(pkl_path):
#         print("No stats.pkl found at", pkl_path)
#         continue
    
#     if DATASET.__eq__('imagenet'):
#         save_dir = f"{SAVE_DIR}/{NET}_{DATASET}_ss{iter}/images/loss/"
#     else:
#         save_dir = f'{SAVE_DIR}/{NET}_{DATASET}/images/loss/'
    
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     acc_save_file = f"{save_dir}/acc.png"
#     loss_save_file = f"{save_dir}/loss.png"

#     with open(pkl_path, 'rb') as f:
#         losses = pickle.load(f)
        
#     X = [loss['epoch'] for loss in losses]

#     '''Create plots of accuracies'''
#     plt.xlabel('Epoch (N)')
#     plt.ylabel('Accuracy')
#     test_acc = [loss['acc_te']/100. for loss in losses]
#     train_acc = [loss['acc_tr']/100. for loss in losses]
#     plt.plot(X, test_acc, label='Test')
#     plt.plot(X, train_acc, label='Train')
#     plt.legend()
#     plt.title('Accuracy v. Epoch')
#     plt.savefig(acc_save_file)
#     plt.clf()

#     '''Create plots of losses'''
#     plt.xlabel('Epoch (N)')
#     plt.ylabel('Loss')
#     test_loss = np.array([np.mean(loss['loss_te']) for loss in losses])
#     test_std = np.array([np.std(loss['loss_te']) for loss in losses])
#     argmin_test_loss = np.argmin(test_loss)
#     min_test_loss = test_loss[argmin_test_loss]
#     plt.fill_between(X, test_loss - test_std, test_loss + test_std, alpha=0.1, interpolate=True)

#     train_loss = np.array([np.mean(loss['loss_tr']) for loss in losses])
#     train_std = np.array([np.std(loss['loss_tr']) for loss in losses])
#     plt.fill_between(X, train_loss - train_std, train_loss + train_std, alpha=0.1, interpolate=True)
#     plt.vlines(X[argmin_test_loss], 0, min_test_loss, linestyles='dashed', label='Min Test Loss')

#     plt.plot(X, test_loss, label='Test Mean')
#     plt.plot(X, train_loss, label='Train Mean')
#     plt.legend()
#     plt.title('Average Loss v. Epoch')
#     plt.savefig(loss_save_file)
#     plt.clf()

''' Concatenate images of curves '''
for iter in range(START, SUBSETS):
    print(f'Processing images for subset {iter}')
    if DATASET.__eq__('imagenet'):
        save_dir = f"{SAVE_DIR}/{NET}_{DATASET}_ss{iter}/images/curves/"
    else:
        save_dir = f'{SAVE_DIR}/{NET}_{DATASET}/images/curves/'
    
    if not os.path.exists(save_dir):
        print("No directory", save_dir)
        continue

    files = os.listdir(save_dir)
    files = [f for f in files if f.startswith('epoch')]

    if len(files) == 0:
        print("No images found at", save_dir)
        continue
    files.sort(key=lambda x: int(x.split('_')[1])) # sort by epoch
    
    images = [Image.open(f"{save_dir}/{f}") for f in files] # open images
    get_concat_v_multi_blank(images).save(f"{save_dir}/concat.png") # concat and save
    