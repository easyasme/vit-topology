import argparse
import os
import pickle
import itertools

import numpy as np
import torch
import plotly.express as px
from gtda.diagrams import Filtering, BettiCurve, PairwiseDistance
from config import UPPER_DIM


parser = argparse.ArgumentParser(description='Post-process diagrams')

parser.add_argument('--net', nargs='+', action='extend', type=str, default=[])
parser.add_argument('--dataset', default=None, type=str, help='Dataset: "mnist" or "imagenet"')
parser.add_argument('--save_dir')
parser.add_argument('--start_iter', default=None, type=int, help='Subset index to start at for imagenet.')
parser.add_argument('--stop_iter', default=None, type=int, help='Subset index to stop at for imagenet.')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', default=[], type=int)
parser.add_argument('--reduction', default=None, type=str, help='Reductions: "pca" or "umap"')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcorr".')

args = parser.parse_args()

NET = args.net # 'lenet', 'alexnet', 'vgg', 'resnet'
DATASET = args.dataset # 'mnist' or 'imagenet'
SAVE_DIR = args.save_dir
START = args.start_iter
STOP = args.stop_iter
EPOCHS = args.chkpt_epochs
RED = args.reduction
METRIC = args.metric

SAVE_DIR = args.save_dir
if len(NET) == 1:
    SAVE_DIR = os.path.join(SAVE_DIR, f'{NET[0]}_{DATASET}_comp')
else:
    SAVE_DIR = os.path.join(SAVE_DIR, f'{NET[0]}_{NET[-1]}_{DATASET}_comp')
print(f'\n ==> Save directory: {SAVE_DIR}')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

EPOCH_COMP_DIR = os.path.join(SAVE_DIR, 'epoch_comp')
if not os.path.exists(EPOCH_COMP_DIR):
    os.makedirs(EPOCH_COMP_DIR)

SS_EPOCH_COMP_DIR = os.path.join(SAVE_DIR, 'subset_epoch_comp')
if not os.path.exists(SS_EPOCH_COMP_DIR):
    os.makedirs(SS_EPOCH_COMP_DIR)

# Initialize device and GTDA transformers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_bins = 100
dgm_filter = Filtering(epsilon=0.02125)
curve = BettiCurve(n_bins=n_bins, n_jobs=-1)

def get_epoch_curves(net, dataset, start, stop):
    curve_dict = {}
    for i in range(start, stop+1):
        print(f'\nProcessing subset {i}:')

        pkl_folder = f'./losses/{net}/{net}_{dataset}_ss{i}' if dataset == 'imagenet' else f'./losses/{net}/{net}_{dataset}'
        pkl_folder += f'/{RED}' if RED is not None else ''
        pkl_folder += f'/{METRIC}' if METRIC is not None else ''

        epoch_dict = {}
        for epoch in EPOCHS:
            print(f'==> Processing epoch {epoch}')

            pkl_fl = os.path.join(pkl_folder, f'dgm_epoch_{epoch}.pkl')
            try:
                with open(pkl_fl, 'rb') as f:
                    dgm_gtda = pickle.load(f)
                dgm_gtda = dgm_filter.fit_transform([dgm_gtda])
                epoch_dict[epoch] = curve.fit_transform(dgm_gtda).squeeze()
            except:
                raise FileNotFoundError(f'Error loading {pkl_fl}')
        curve_dict[i] = epoch_dict
    
    return curve_dict

def compute_net_epoch_distances(epoch_dict_1, epoch_dict_2, start, stop, epochs, permute=True):
    ''' Compute pairwise distances across epochs for networks using the same subsets;
    epoch_dict_1: dictionary of betti curves for each epoch and subset
    epoch_dict_2: dictionary of betti curves for each epoch and subset
    start: start subset index
    stop: stop subset index
    epochs: list of epochs to compute distances for
    return: list of pairwise distances across networks for each subset
    '''
    dist_list = []
    for i in range(start, stop+1):
        dgm_1 = np.array([epoch_dict_1[i][epoch] for epoch in epochs], dtype=np.float32)
        dgm_2 = np.array([epoch_dict_2[i][epoch] for epoch in epochs], dtype=np.float32)
        dgm_1 = torch.tensor(dgm_1).to(device).permute(1,0,2) if permute else torch.tensor(dgm_1).to(device)
        dgm_2 = torch.tensor(dgm_2).to(device).permute(1,0,2) if permute else torch.tensor(dgm_2).to(device)

        dist_list.append(torch.cdist(dgm_1, dgm_2, p=np.inf).numpy(force=True))

        del dgm_1, dgm_2
        torch.cuda.empty_cache()
    
    return dist_list

def compute_net_subset_distances(epoch_dict_1, epoch_dict_2, start, stop, epochs, permute=True):
    ''' Compute pairwise distances across networks using the same subsets and epochs;
    epoch_dict_1: dictionary of betti curves for each epoch and subset
    epoch_dict_2: dictionary of betti curves for each epoch and subset
    start: start subset index
    stop: stop subset index
    epochs: list of epochs to compute distances for
    return: list of pairwise distances across networks for each epoch
    '''
    dist_list_ss = []
    for epoch in epochs:
        dgm_1 = np.array([epoch_dict_1[i][epoch] for i in range(start, stop+1)], dtype=np.float32)
        dgm_2 = np.array([epoch_dict_2[i][epoch] for i in range(start, stop+1)],dtype=np.float32)
        dgm_1 = torch.tensor(dgm_1).to(device).permute(1,0,2) if permute else torch.tensor(dgm_1).to(device)
        dgm_2 = torch.tensor(dgm_2).to(device).permute(1,0,2) if permute else torch.tensor(dgm_2).to(device)

        dist_list_ss.append(torch.cdist(dgm_1, dgm_2, p=np.inf).numpy(force=True))

        del dgm_1, dgm_2
        torch.cuda.empty_cache()
    
    return dist_list_ss

def vis_across_epochs(dist_list, two_nets=False):
    ''' Visualize the distances across epochs for every subset for each net's distances in dist_lists '''
    mean_dist_mat = np.mean(np.array(dist_list), axis=0)
    for dim in range(UPPER_DIM+1):
        mean_fig = px.imshow(mean_dist_mat[dim],
                        labels=dict(x=f'{NET[0]} epochs', y=f'{NET[-1]} epochs', color='Distance'),
                        title=f'Average pairwise distances across subsets in homology dimension {dim}',
                        x=[str(epoch) for epoch in EPOCHS],
                        y=[str(epoch) for epoch in EPOCHS],
                        width=800,
                        height=800)
        filename = f'net_' if two_nets else ''
        filename += f'avg_dist_dim{dim}.png'
        mean_fig.write_image(os.path.join(EPOCH_COMP_DIR, filename), format='png')
        
        for i,dist_mat in enumerate(dist_list):
            fig = px.imshow(dist_mat[dim],
                            labels=dict(x=f'{NET[0]} epochs', y=f'{NET[-1]} epochs', color='Distance'),
                            title=f'Pairwise distances across epochs for subset {i} in homology dimension {dim}',
                            x=[str(epoch) for epoch in EPOCHS],
                            y=[str(epoch) for epoch in EPOCHS],
                            width=800,
                            height=800)
            filename = f'net_' if two_nets else ''
            filename += f'dist_ss{i}_dim{dim}.png'
            fig.write_image(os.path.join(EPOCH_COMP_DIR, filename), format='png')

def vis_across_subsets(dist_list_ss, two_nets=False):
    mean_dist_mat = np.mean(np.array(dist_list_ss), axis=0)
    for dim in range(UPPER_DIM+1):
        mean_fig = px.imshow(mean_dist_mat[dim],
                        labels=dict(x=f'{NET[0]} subsets', y=f'{NET[-1]} subsets', color='Distance'),
                        title=f'Average pairwise distances across epochs in homology dimension {dim}',
                        width=900,
                        height=900)
        filename = f'net_' if two_nets else ''
        filename += f'avg_dist_dim{dim}.png'
        mean_fig.write_image(os.path.join(SS_EPOCH_COMP_DIR, filename), format='png')
        
        for i,dist_mat in enumerate(dist_list_ss):
            fig = px.imshow(dist_mat[dim],
                            labels=dict(x=f'{NET[0]} subsets', y=f'{NET[-1]} subsets', color='Distance'),
                            title=f'Pairwise distances across subsets for epoch {EPOCHS[i]} in homology dim {dim}',
                            width=900,
                            height=900)
            filename = f'net_' if two_nets else ''
            filename += f'dist_epoch{EPOCHS[i]}_dim{dim}.png'
            fig.write_image(os.path.join(SS_EPOCH_COMP_DIR, filename), format='png')


# Load the betti curves
net_dict_list = []
for net in NET:
    # each dictionary contains betti curves for each epoch in a specific subset
    print(f'\nProcessing network {net}')
    net_dict_list.append(get_epoch_curves(net, DATASET, START, STOP))

# Compute pairwise distances across epochs for different networks using the same subsets
print(f'\nProcessing networks {NET[0]} and {NET[-1]} across epochs')
dist_list = compute_net_epoch_distances(net_dict_list[0], net_dict_list[-1], START, STOP, EPOCHS)
    
print(f'\nProcessing networks {NET[0]} and {NET[-1]} across subsets')
dist_list_ss = compute_net_subset_distances(net_dict_list[0], net_dict_list[-1], START, STOP, EPOCHS)

# Make visualizations
vis_across_epochs(dist_list, two_nets=(len(NET) > 1))
vis_across_subsets(dist_list_ss, two_nets=(len(NET) > 1))
