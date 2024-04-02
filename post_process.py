import argparse
import os
import pickle

import numpy as np
import plotly.express as px
from gtda.curves import Derivative
from gtda.diagrams import BettiCurve, PersistenceEntropy, PersistenceImage
from gtda.plotting import plot_betti_curves, plot_betti_surfaces, plot_diagram

from config import UPPER_DIM

parser = argparse.ArgumentParser(description='Post-process Betti numbers')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_dir')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--iter', default=0, type=int)

args = parser.parse_args()

NET = args.net # 'lenet', 'alexnet', 'vgg', 'resnet', 'densenet'
DATASET = args.dataset # 'mnist' or 'imagenet'
SAVE_DIR = args.save_dir
EPOCHS = args.chkpt_epochs

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

if DATASET == 'imagenet':
    pkl_folder = f'./losses/{NET}/{NET}_{DATASET}_ss{args.iter}'
else:
    pkl_folder = f'./losses/{NET}/{NET}_{DATASET}'

# Initialize GTDA transformers
n_bins = 100
curve = BettiCurve(n_bins=n_bins, n_jobs=-1)
pers_entropy = PersistenceEntropy(n_jobs=-1)
pers_img = PersistenceImage(n_bins=n_bins, n_jobs=-1)

# Load the persistence diagrams
samplings = np.linspace(0, 1, n_bins)
samplings = np.tile(samplings, (UPPER_DIM+1,1))

curves_list = []
dgm_list = []
for epoch in EPOCHS:
    print(f'Processing epoch {epoch}')

    pkl_fl = os.path.join(pkl_folder, f'dgm_epoch_{epoch}.pkl')
    try:
        with open(pkl_fl, 'rb') as f:
            dgm_gtda = pickle.load(f)
    except:
        raise FileNotFoundError(f'Error loading {pkl_fl}')
    
    nodes = len(dgm_gtda[:,0])

    # Compute statistics
    betti_curves = curve.fit_transform([dgm_gtda])
    img = pers_img.fit_transform([dgm_gtda])

    curves_list.append(betti_curves[0])
    dgm_list.append(dgm_gtda)

    # Plot persistence diagrams
    diagram = plot_diagram(dgm_gtda)
    diagram.write_image(os.path.join(PERS_DIR, f'epoch_{epoch}_diag.png'), format='png')

    # Plot Betti curves
    curve_dict = {"title": f'Epoch {epoch}', "x": '$\epsilon$', "y": 'Betti number (N)'}
    betti_curve = plot_betti_curves(betti_curves[0]/nodes, samplings=samplings, homology_dimensions=range(1, UPPER_DIM+1), plotly_params=curve_dict)
    betti_curve.write_image(os.path.join(CURVES_DIR, f'epoch_{epoch}_curve.png'), format='png')

    # Plot persistence images
    for i in range(1, UPPER_DIM+1):
        img_fig = pers_img.plot(img, homology_dimension_idx=i, plotly_params=curve_dict)
        img_fig.write_image(os.path.join(PERS_DIR, f'epoch_{epoch}_pers_img_{i}.png'), format='png')

curves_list = np.array(curves_list)
# dgm_list = np.array(dgm_list)

diff = np.diff(curves_list, n=1, axis=0) # compute the first difference between consecutive epochs
average_diff = np.mean(diff, axis=0)

# Plot average diff Betti curve
diff_dict = {"title": 'Average Diff. Betti curve', "x": '$\epsilon$', 'y': 'Betti number (N)'}
for i in range(UPPER_DIM+1):
    px.line(x=samplings[0], y=average_diff[i], title='Average Diff Betti curve', labels={'x': '$\epsilon$', 'y': 'Betti number'}).write_image(os.path.join(DIFF_DIR, f'avg_diff_curve_dim_{i}.png'), format='png')

# Plot Betti surface
betti_surface = plot_betti_surfaces(curves_list, samplings=samplings)
for i, fig in enumerate(betti_surface):
    fig.update_layout(
        grid_xaxes=[str(epoch) for epoch in EPOCHS],
        scene=dict(
            xaxis_title='$\epsilon$',
            yaxis_title='Time (Epochs)',
            zaxis_title='Betti number (N)'
            )
        )
    fig.write_image(os.path.join(SURF_DIR, f'surf_dim_{i}.png'), format='png')

# Plot persistence entropy
# entropy = pers_entropy.fit_transform(dgm_list)
# fig = px.line(title='Persistence entropies', labels={'x': 'Homology dimension', 'y': 'Entropy'})
# for dim in range(entropy.shape[1]):
#     fig.add_scatter(y=entropy[:, dim], name=f"PE in homology dimension {dim}")
#     fig.write_image(os.path.join(PERS_DIR, f'entropy_epoch_{epoch}_dim_{dim}.png'), format='png')

# python post_process.py --net lenet --dataset mnist --chkpt_epochs 0 4 8 20 30 40 50 --save_dir ./results/test/lenet_mnist