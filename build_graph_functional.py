from __future__ import print_function

import argparse
import time
import numpy as np

from gtda.homology import VietorisRipsPersistence
from gtda.utils import check_diagrams
from scipy.sparse import coo_matrix

from config import UPPER_DIM, POS_EMB_FLAG
from graph import *
from loaders import *
from models.utils import get_model
from passers import Passer
from utils import *


parser = argparse.ArgumentParser(description='Build Graph and Compute Betti Numbers')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--subset', default=50, type=int, help='Subset size for building graph.')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: none, spearman, dcorr, or callable.')
parser.add_argument('--average', default=0, type=int, help='Average over all samples.')
parser.add_argument('--reduction', default=None, type=str, help='Reductions: pca, umap, kmeans or cla.')
parser.add_argument('--exp', default=1, type=float, help='Exponent for correlation distance.')

args = parser.parse_args()

device_list = get_device_list()

''' Directory to save persistence diagrams '''
pkl_folder = f'./dgms/{args.net}/{args.net}_{args.dataset}' if POS_EMB_FLAG else f'./dgms/no_pe/{args.net}/{args.net}_{args.dataset}'
pkl_folder += f'/{args.reduction}' if args.reduction is not None else ''
pkl_folder += f'/{args.metric}' if args.metric is not None else ''
pkl_folder += f'/{args.subset}' if args.subset is not None else ''

''' Create save directories to correlations '''
corr_folder = f'./corrs/{args.net}/{args.net}_{args.dataset}' if POS_EMB_FLAG else f'./corrs/no_pe/{args.net}/{args.net}_{args.dataset}'
corr_folder += f'/{args.reduction}' if args.reduction is not None else ''
corr_folder += f'/{args.metric}' if args.metric is not None else ''
corr_folder += f'/{args.subset}' if args.subset is not None else ''

# Build models
print('\n ==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device_list[0])
net.eval()

''' Prepare test data loader '''
test_loader = loader(f'{args.dataset}_test', batch_size=100, subset=args.subset, transform=net._get_transform())

''' Instantiate passer '''
passer = Passer(net, test_loader, None, device_list[0])

''' Instatiate Vietoris-Rips '''
rips = VietorisRipsPersistence(metric='precomputed',
                               homology_dimensions=[int(i) for i in range(UPPER_DIM + 1)],
                               collapse_edges=True,
                               infinity_values=np.inf,
                               reduced_homology=True,
                               n_jobs=-1)

with torch.no_grad():
    ''' Get activations and reduce dimensionality '''
    activs = passer.get_function(reduction=args.reduction,
                                 device_list=device_list,
                                 corr=args.metric if not None else 'pearson',
                                 exp=args.exp,
                                 average=args.average) 
    adjs = adjacency(activs, metric=args.metric, device=device_list[0]) # compute distance adjacency matrix

    corr_pkl_file = os.path.join(corr_folder, f'corr.pkl')
    if not os.path.exists(os.path.dirname(corr_pkl_file)):
        os.makedirs(os.path.dirname(corr_pkl_file))
    with open(corr_pkl_file, 'wb') as f:
        pickle.dump(adjs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # convert to distance matrix for V-R filtration
    c = 1
    adjs = [1 - c * torch.arccos(x)/torch.pi for x in adjs]

    # convert to COO format for faster computation of persistence diagram
    coo_adjs = []
    for adj in adjs:
        indices = (adj<=2.).nonzero().numpy(force=True)
        i = list(zip(*indices))
        vals = adj[i].flatten().numpy(force=True)

        adj = coo_matrix((vals, i), shape=adj.shape) if len(i) > 0 else coo_matrix(([], ([], [])), shape=adj.shape)

        coo_adjs.append(adj)

    print(f'\nThe length of the COO distance matrix is {len(coo_adjs)}')

    del indices, i, vals, adjs, activs

    # Compute persistence diagram
    comp_time = time.time()
    dgm = rips.fit_transform(coo_adjs)
    dgm = check_diagrams(dgm)
    comp_time = time.time() - comp_time
    print(f'\n PH computation time: {comp_time/60:.2f} minutes \n')

    # free GPU memory
    del coo_adjs
    torch.cuda.empty_cache()

    # Save persistence diagrams
    dgm_pkl_file = os.path.join(pkl_folder, f'dgm.pkl')
    if not os.path.exists(os.path.dirname(dgm_pkl_file)):
        os.makedirs(os.path.dirname(dgm_pkl_file))
    with open(dgm_pkl_file, 'wb') as f:
        pickle.dump([dgm, comp_time], f, protocol=pickle.HIGHEST_PROTOCOL)

    del dgm, comp_time

del passer, net, test_loader
