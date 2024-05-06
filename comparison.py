import argparse
import os
import pickle

import numpy as np
import plotly.express as px
from gtda.diagrams import Filtering, BettiCurve, PairwiseDistance
from config import UPPER_DIM


parser = argparse.ArgumentParser(description='Post-process diagrams')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_dir')
parser.add_argument('--start_iter', default=None, type=int, help='Subset index to start at.')
parser.add_argument('--stop_iter', default=None, type=int, help='Subset index to stop at.')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--reduction', default=None, type=str, help='Reductions: "pca" or "umap"')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcorr", or callable.')

args = parser.parse_args()

NET = args.net # 'lenet', 'alexnet', 'vgg', 'resnet', 'densenet'
DATASET = args.dataset # 'mnist' or 'imagenet'
SAVE_DIR = args.save_dir
START = args.start_iter
STOP = args.stop_iter
EPOCHS = args.chkpt_epochs
RED = args.reduction
METRIC = args.metric
PH_METRIC = 'bottleneck' # 'betti', 'wasserstein', 'bottleneck'

SAVE_DIR = args.save_dir
print(f'\n ==> Save directory: {SAVE_DIR}')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

EPOCH_COMP_DIR = os.path.join(SAVE_DIR, 'epoch_comp')
if not os.path.exists(EPOCH_COMP_DIR):
    os.makedirs(EPOCH_COMP_DIR)

SS_EPOCH_COMP_DIR = os.path.join(SAVE_DIR, 'subset_epoch_comp')
if not os.path.exists(SS_EPOCH_COMP_DIR):
    os.makedirs(SS_EPOCH_COMP_DIR)

# Initialize GTDA transformers
n_bins = 100
dgm_filter = Filtering(epsilon=0.025)#125)
curve = BettiCurve(n_bins=n_bins, n_jobs=-1)

if PH_METRIC == 'betti':
    metric_params = {'p': 2., 'n_bins': n_bins}
elif PH_METRIC == 'wasserstein':
    metric_params = {'p': 2., 'delta': 0.01} # p for L^p distance; delta=0 not implemented
elif PH_METRIC == 'bottleneck':
    metric_params = {'delta': 0.} # set delta>0 for approximate bottleneck distance
dist = PairwiseDistance(metric=PH_METRIC, metric_params=metric_params, order=None, n_jobs=-1)

# Load the persistence diagrams
epoch_dict = {}
for i in range(START, STOP+1):
    print(f'\nProcessing subset {i}:')

    pkl_folder = f'./losses/{NET}/{NET}_{DATASET}_ss{i}' if DATASET == 'imagenet' else f'./losses/{NET}/{NET}_{DATASET}'
    pkl_folder += f'/{RED}' if RED is not None else ''
    pkl_folder += f'/{METRIC}' if METRIC is not None else ''

    dgm_list = []
    for epoch in EPOCHS:
        print(f'==> Processing epoch {epoch}')

        # dgm_gtda = None
        pkl_fl = os.path.join(pkl_folder, f'dgm_epoch_{epoch}.pkl')
        try:
            with open(pkl_fl, 'rb') as f:
                dgm_gtda = pickle.load(f)
            dgm_gtda = dgm_filter.fit_transform([dgm_gtda]) if dgm_gtda is not None else None
            dgm_list.append(dgm_gtda.squeeze())
        except:
            raise FileNotFoundError(f'Error loading {pkl_fl}')
    
    # Find the minimum number of points for each dimension for each epoch
    min_pts = []
    for dim in range(UPPER_DIM+1):
        minimum = min([sum(dgm[:,[-1]]==dim) for dgm in dgm_list]).item()
        min_pts.append(minimum)
    print(f'\nMinimum number of points for each dimension: {min_pts}\n')

    for j,dgm in enumerate(dgm_list):
        reduced_list = []
        for minimum, dim in zip(min_pts, range(UPPER_DIM+1)):
            temp = dgm[(dgm[:,[-1]]==dim).flatten(),:]
            temp_indices = np.random.choice(range(len(temp)), minimum, replace=False)
            temp = temp[temp_indices, :]
            reduced_list.append(temp)
        dgm_list[j] = np.concatenate(reduced_list, axis=0)
    epoch_dict[i] = {epoch: dgm for epoch, dgm in zip(EPOCHS, dgm_list)}

# Compute pairwise distances across epochs using the same network and subset
dist_list = []
for i in range(START, STOP+1):
    if DATASET == 'imagenet':
        print(f'Computing distances for {DATASET} subset {i}')
    else:
        print(f'\nComputing distances for {DATASET}\n')
    dgm_list = [epoch_dict[i][epoch] for epoch in EPOCHS]
    dist_list.append(dist.fit_transform(dgm_list))
    
# Compute pairwise distances across subsets using the same network and epoch
dist_list_ss = []
for epoch in EPOCHS:
    print(f'Computing distances for epoch {epoch}')
    dgm_list = [epoch_dict[i][epoch] for i in range(START, STOP+1)]
    dist_list_ss.append(dist.fit_transform(dgm_list))

# Visualize the distances across epochs
for dim in range(UPPER_DIM+1):
    dist_mat = []
    for i,mat in enumerate(dist_list):
        dist_mat.append(mat[:,:,[dim]])
    dist_mat = np.concatenate(dist_mat, axis=-1).squeeze()
    fig = px.imshow(dist_mat, labels=dict(x='Epoch', y='Epoch', color='Distance'), title=f'Pairwise distances across epochs for subset {i}', width=800, height=800)
    fig.write_image(os.path.join(EPOCH_COMP_DIR, f'distances_ss{0}_dim{dim}.png'), format='png')

# # Visualize the distances across subsets
# for dim in range(UPPER_DIM+1):
#     dist_mat = []
#     for i,mat in enumerate(dist_list_ss):
#         dist_mat.append(mat[:,:,[dim]])
#     dist_mat = np.concatenate(dist_mat, axis=-1).squeeze()
#     fig = px.imshow(dist_mat, labels=dict(x='Subset', y='Subset', color='Distance'), title=f'Pairwise distances across subsets for epoch {EPOCHS[i]}', width=800, height=800)
#     fig.write_image(os.path.join(SS_EPOCH_COMP_DIR, f'distances_epoch{EPOCHS[i]}.png'), format='png')
