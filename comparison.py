import argparse
import os
import pickle

import numpy as np
import plotly.express as px
from gtda.diagrams import BettiCurve, PairwiseDistance
from config import UPPER_DIM


parser = argparse.ArgumentParser(description='Post-process diagrams')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_dir')
parser.add_argument('--start_ss', default=None, type=int, help='Subset index to start at.')
parser.add_argument('--stop_ss', default=None, type=int, help='Subset index to stop at.')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--reduction', default=None, type=str, help='Reductions: "pca" or "umap"')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcorr", or callable.')

args = parser.parse_args()

NET = args.net # 'lenet', 'alexnet', 'vgg', 'resnet', 'densenet'
DATASET = args.dataset # 'mnist' or 'imagenet'
SAVE_DIR = args.save_dir
START = args.start_ss
STOP = args.stop_ss
EPOCHS = args.chkpt_epochs
RED = args.reduction
METRIC = args.metric
ITER = args.iter

''' Create save directories to store images '''
SAVE_DIR = args.save_dir
print(f'\n ==> Save directory: {SAVE_DIR} \n')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

EPOCH_COMP_DIR = os.path.join(SAVE_DIR, 'epoch_comp')
if not os.path.exists(EPOCH_COMP_DIR):
    os.makedirs(EPOCH_COMP_DIR)

SS_EPOCH_COMP_DIR = os.path.join(SAVE_DIR, 'subset_epoch_comp')
if not os.path.exists(SS_EPOCH_COMP_DIR):
    os.makedirs(SS_EPOCH_COMP_DIR)

CURVES_DIR = os.path.join(IMG_DIR, 'curves')
if not os.path.exists(CURVES_DIR):
    os.makedirs(CURVES_DIR)

DIFF_DIR = os.path.join(IMG_DIR, 'diff')
if not os.path.exists(DIFF_DIR):
    os.makedirs(DIFF_DIR)

SURF_DIR = os.path.join(IMG_DIR, 'surfaces')
if not os.path.exists(SURF_DIR):
    os.makedirs(SURF_DIR)

ENT_DIR = os.path.join(IMG_DIR, 'entropy')
if not os.path.exists(ENT_DIR):
    os.makedirs(ENT_DIR)

pkl_folder = f'./losses/{NET}/{NET}_{DATASET}_ss{ITER}' if DATASET == 'imagenet' else f'./losses/{NET}/{NET}_{DATASET}'
pkl_folder += f'/{RED}' if RED is not None else ''
pkl_folder += f'/{METRIC}' if METRIC is not None else ''

# Initialize GTDA transformers
n_bins = 100
dgm_filter = Filtering(epsilon=0.02125)
curve = BettiCurve(n_bins=n_bins, n_jobs=-1)
dist = PairwiseDistance(metric='betti', metric_params={}, order=None, n_jobs=None)


# Load the persistence diagrams
samplings = np.linspace(0, 1, n_bins)
samplings = np.tile(samplings, (UPPER_DIM+1,1))

curves_list = []
dgm_list = []
for epoch in EPOCHS:
    print(f'Processing epoch {epoch}')

    dgm_gtda = None
    pkl_fl = os.path.join(pkl_folder, f'dgm_epoch_{epoch}.pkl')
    try:
        with open(pkl_fl, 'rb') as f:
            dgm_gtda = pickle.load(f)
        dgm_gtda = dgm_filter.fit_transform([dgm_gtda]) if dgm_gtda is not None else None
        dgm_list.append(dgm_gtda)
    except:
        raise FileNotFoundError(f'Error loading {pkl_fl}')
    import argparse
