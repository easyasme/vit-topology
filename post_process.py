import argparse
import os
import pickle

import numpy as np
import plotly.express as px
from gtda.curves import Derivative
from gtda.diagrams import BettiCurve, PersistenceEntropy, PersistenceImage, Filtering
from gtda.plotting import plot_betti_curves, plot_betti_surfaces, plot_diagram

from config import UPPER_DIM

parser = argparse.ArgumentParser(description='Post-process diagrams')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_dir')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--reduction', default=None, type=str, help='Reductions: "pca" or "umap"')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcorr", or callable.')
parser.add_argument('--iter', default=0, type=int)

args = parser.parse_args()

NET = args.net # 'lenet', 'alexnet', 'vgg', 'resnet', 'densenet'
DATASET = args.dataset # 'mnist' or 'imagenet'
SAVE_DIR = args.save_dir
EPOCHS = args.chkpt_epochs
RED = args.reduction
METRIC = args.metric
ITER = args.iter

''' Create save directories to store images '''
SAVE_DIR = args.save_dir
print(f'\n ==> Save directory: {SAVE_DIR} \n')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

IMG_DIR = os.path.join(SAVE_DIR, 'images')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

PERS_DIR = os.path.join(IMG_DIR, 'persistence')
if not os.path.exists(PERS_DIR):
    os.makedirs(PERS_DIR)

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
pers_entropy = PersistenceEntropy(n_jobs=-1)

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
    
    nodes = len([x for x in dgm_gtda[0] if x[-1] == 0]) if dgm_gtda[0] is not None else 1

    # Compute statistics
    betti_curves = curve.fit_transform(dgm_gtda) if dgm_gtda[0] is not None else None
    if betti_curves is not None:
        curves_list.append(betti_curves[0]/nodes)

    # Plot persistence diagrams
    if dgm_gtda[0] is not None:
        diagram = plot_diagram(dgm_gtda[0])
        diagram.write_image(os.path.join(PERS_DIR, f'epoch_{epoch}_diag.png'), format='png')

    # Plot Betti curves
    if betti_curves is not None:
        betti_curve = plot_betti_curves(betti_curves[0]/nodes, samplings=samplings, homology_dimensions=range(1, UPPER_DIM+1))
        betti_curve.update_layout(title=f'Epoch {epoch}',
                                scene=dict(
                                        xaxis_title='Filtering parameter',
                                        yaxis_title='Betti number/node (N)',
                                        ),
                                )
        betti_curve.write_image(os.path.join(CURVES_DIR, f'epoch_{epoch}_curve.png'), format='png')

# Convert list to numpy array
curves_list = np.array(curves_list) if len(curves_list) > 0 else None

# Compute the average difference between consecutive epochs
avg_diff = np.diff(curves_list, n=1, axis=0).mean(axis=0) if curves_list is not None else None

# Plot average diff Betti curve
if avg_diff is not None:
    diff_dict = {"title": 'Average Diff. Betti curve', "x": '$\epsilon$', 'y': 'Betti number (N)'}
    for i in range(UPPER_DIM+1):
        px.line(x=samplings[0], y=avg_diff[i], title='Average Diff Betti curve', labels={'x': '$\epsilon$', 'y': 'Betti number'}).write_image(os.path.join(DIFF_DIR, f'avg_diff_curve_dim_{i}.png'), format='png')

# Plot Betti surface
if curves_list is not None:
    betti_surface = plot_betti_surfaces(curves_list, samplings=samplings)
    for i, fig in enumerate(betti_surface):
        fig.update_layout(
            scene=dict(
                xaxis_title='Filtering parameter',
                yaxis_title='Time (Epochs)',
                zaxis_title='Betti number (N)',
                camera_eye=dict(x=1.5, y=1.5, z=1.25),
                yaxis=dict(tickmode='array', tickvals=list(range(len(EPOCHS))), ticktext=[str(epoch) for epoch in EPOCHS]),
                )
            )
        fig.write_image(os.path.join(SURF_DIR, f'surf_dim_{i}.png'), format='png')

# Compute persistence entropy
if len(dgm_list) != 0:
    entropy_list =[]
    for dgm in dgm_list:
        entropy = pers_entropy.fit_transform(dgm) if dgm[0] is not None else None
        if entropy is not None:
            entropy_list.append(entropy[0])

    entropy_list = np.array(entropy_list) if len(entropy_list) > 0 else None
    
    # Plot persistence entropy
    if entropy_list is not None:
        for i in range(UPPER_DIM+1):
            px.line(x=EPOCHS, y=entropy_list[:,i], title=f'Persistence entropy dimension {i}', labels={'x': 'Epoch', 'y': 'Entropy'}).write_image(os.path.join(ENT_DIR, f'entropy_dim_{i}.png'), format='png')
