import argparse
import os
import pickle

import numpy as np
import plotly.express as px
from gtda.curves import Derivative
from gtda.diagrams import BettiCurve, Filtering
from gtda.plotting import plot_betti_curves, plot_betti_surfaces, plot_diagram

from config import UPPER_DIM

parser = argparse.ArgumentParser(description='Post-process diagrams')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_dir', default='./results', help='Directory to save results.')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--reduction', default=None, type=str, help='Reductions: pca, kmeans, umap, or cla.')
parser.add_argument('--subset', default=50, type=int, help='Subset of the dataset to use.')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: spearman, dcorr, or callable.')

args = parser.parse_args()

NET = args.net # 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32'
DATASET = args.dataset # 'imagenet'
SAVE_DIR = args.save_dir
RED = args.reduction
METRIC = args.metric
SUBSET = args.subset

''' Create save directories to store images '''
SAVE_DIR = args.save_dir
print(f'\n ==> Save directory: {SAVE_DIR} \n')

if RED == 'cla':  
    SAVE_DIR = os.path.join(SAVE_DIR, 'cla')
    
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

IMG_DIR = os.path.join(SAVE_DIR, f'images_{SUBSET}')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

PERS_DIR = os.path.join(IMG_DIR, 'persistence')
if not os.path.exists(PERS_DIR):
    os.makedirs(PERS_DIR)

CURVES_DIR = os.path.join(IMG_DIR, 'curves')
if not os.path.exists(CURVES_DIR):
    os.makedirs(CURVES_DIR)

SURF_DIR = os.path.join(IMG_DIR, 'surfaces')
if not os.path.exists(SURF_DIR):
    os.makedirs(SURF_DIR)


pkl_folder = f'./dgms/{NET}/{NET}_{DATASET}'
pkl_folder += f'/{RED}' if RED is not None else ''
pkl_folder += f'/{METRIC}' if METRIC is not None else ''
pkl_folder += f'/{args.subset}' if args.subset is not None else ''

print(f' ==> Persistence diagram directory: {pkl_folder} \n')

# Initialize GTDA transformers
n_bins = 100
dgm_filter = Filtering(epsilon=0.02125)
curve = BettiCurve(n_bins=n_bins, n_jobs=-1)

# Load the persistence diagrams
samplings = np.linspace(0, 1, n_bins)
samplings = np.tile(samplings, (UPPER_DIM+1,1))

curves_list = []
dgm_list = []
dgm_gtda = None
    
pkl_fl = os.path.join(pkl_folder, f'dgm.pkl')
try:
    with open(pkl_fl, 'rb') as f:
        dgm_gtda = pickle.load(f)[0]
except:
    raise FileNotFoundError(f'Error loading {pkl_fl}')

# dgm_gtda = dgm_filter.fit_transform(dgm_gtda) if dgm_gtda is not None else None

# Compute statistics
betti_curves = curve.fit_transform(dgm_gtda) if dgm_gtda is not None else None
for curve in betti_curves:
    if curve is not None:
        curves_list.append(curve)

# Plot persistence diagrams
for i, dgm in enumerate(dgm_gtda):
    if dgm is not None:
        diagram = plot_diagram(dgm)
        diagram.write_image(os.path.join(PERS_DIR, f'encoder_block{i}.png'), format='png')

# Plot Betti curves
if betti_curves is not None:
    for i,curve in enumerate(betti_curves):
        betti_curve = plot_betti_curves(curve, samplings=samplings, homology_dimensions=range(1, UPPER_DIM+1))
        betti_curve.update_layout(title=f'Encoder layer',
                                scene=dict(
                                    xaxis_title='Filtering parameter',
                                    yaxis_title='Betti number/node (N)',
                                ),
                            )
        betti_curve.write_image(os.path.join(CURVES_DIR, f'curve_encoder{i}.png'), format='png')

# Convert list to numpy array
curves_list = np.array(curves_list) if len(curves_list) > 0 else None

# Plot Betti surface
if curves_list is not None:
    betti_surface = plot_betti_surfaces(curves_list, samplings=samplings)
    for i, fig in enumerate(betti_surface):
        fig.update_layout(
                scene=dict(
                    xaxis_title='Filtering parameter',
                    yaxis_title='Depth (encoder layer)',
                    zaxis_title='Betti number (N)',
                    camera_eye=dict(x=1.5, y=1.5, z=1.25),
                    yaxis=dict(tickmode='array',
                           tickvals=list(range(len(dgm_gtda))),
                           ticktext=[str(layer) for layer in range(len(dgm_gtda))]
                    )
                )
            )
        fig.write_image(os.path.join(SURF_DIR, f'surf_dim_{i}.png'), format='png')
