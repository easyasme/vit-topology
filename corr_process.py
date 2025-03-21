import argparse
import os
import pickle

import numpy as np
import plotly.express as px

from config import POS_EMB_FLAG


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
SAVE_DIR = os.path.join(SAVE_DIR, f'{NET}_{DATASET}')
print(f'\n ==> Save directory: {SAVE_DIR} \n')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

IMG_DIR = os.path.join(SAVE_DIR, f'images_{SUBSET}')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

CORR_DIR = os.path.join(IMG_DIR, 'correlation')
if not os.path.exists(CORR_DIR):
    os.makedirs(CORR_DIR)

corr_folder = f'./corrs/{NET}/{NET}_{DATASET}' if POS_EMB_FLAG else f'./corrs/no_pe/{NET}/{NET}_{DATASET}'
corr_folder += f'/{RED}' if RED is not None else ''
corr_folder += f'/{METRIC}' if METRIC is not None else ''
corr_folder += f'/{args.subset}' if args.subset is not None else ''

# Load the correlation matrices
pkl_fl = os.path.join(corr_folder, f'corr.pkl')
try:
    with open(pkl_fl, 'rb') as f:
        corrs = pickle.load(f)
except:
    raise FileNotFoundError(f'Error loading {pkl_fl}')

# Plot correlation heatmaps
for i,corr in enumerate(corrs):
    corr = corr.numpy(force=True)

    fig = px.imshow(corr,
                labels=dict(color="Value"),
                x=list(range(len(corr))),
                y=list(range(len(corr))),
                color_continuous_scale="Viridis") # You can change the color scale

    # Customize the layout (optional)
    fig.update_layout(title=f'Encoder Block {i}')

    # Save the figure
    image_path = os.path.join(CORR_DIR, f'encoder_{i}.png')
    fig.write_image(image_path)
    
    print(f'==> Saved {image_path}')

# Plot average over dataset subsets
corr_dict = {i: [] for i in range(50, 500+50, 50)}
for i in range(50, 500+50, 50):
    corr_folder = f'./corrs/{NET}/{NET}_{DATASET}'
    corr_folder += f'/{RED}' if RED is not None else ''
    corr_folder += f'/{METRIC}' if METRIC is not None else ''
    corr_folder += f'/{i}'

    # Load the correlation matrices
    pkl_fl = os.path.join(corr_folder, f'corr.pkl')
    try:
        with open(pkl_fl, 'rb') as f:
            corrs = pickle.load(f)
    except:
        raise FileNotFoundError(f'Error loading {pkl_fl}')

    for corr in corrs:
        corr = corr.numpy(force=True)
        off_avg = np.mean(corr) - np.trace(corr) / corr.size
        corr_dict[i].append(off_avg) # calculate avg correlation without 

# Plot the average correlation over the dataset subsets
X = list(range(50, 500+50, 50))
for i in range(len(corr_dict[50])):
    Y = [corr_dict[j][i] for j in range(50, 500+50, 50)]
        
    fig = px.line(x=X, y=Y, title=f'Average correlation over dataset subsets for encoder block {i}')
    fig.update_layout(xaxis_title='Subset size',
                      yaxis_title='Average correlation',
                      xaxis=dict(tickmode='linear', tick0=50, dtick=50),
                      yaxis=dict(tickmode='linear', tick0=0.0, dtick=0.1),
                      yaxis_range=[0.0, 1.0])
    
    image_path = os.path.join(CORR_DIR, f'encoder_{i}_avg.png')
    fig.write_image(image_path)
        
    print(f'==> Saved avg plots {image_path}')
