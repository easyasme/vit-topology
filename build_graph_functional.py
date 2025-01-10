from __future__ import print_function

import argparse
import time

from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams
from gtda.utils import check_diagrams
from scipy.sparse import coo_matrix

from bettis import betti_nums
from config import UPPER_DIM, SEED
from graph import *
from loaders import *
from models.utils import get_model
from passers import Passer
from utils import *

import numpy as np
import random


parser = argparse.ArgumentParser(description='Build Graph and Compute Betti Numbers')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--subset', default=500, type=int, help='Subset size for building graph.')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: none, spearman, dcorr, or callable.')
parser.add_argument('--average', default=1, type=int, help='Average over all samples.')
parser.add_argument('--thresholds', default='0. 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--eps_thresh', default=1., type=float)
parser.add_argument('--reduction', default=None, type=str, help='Reductions: pca, umap, kmeans or cla.')
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=20, type=int, help='resume from epoch')
parser.add_argument('--exp', default=1, type=float, help='Exponent for correlation distance.')
parser.add_argument('--it', default=0, type=int)
parser.add_argument('--verbose', default=0, type=int)

args = parser.parse_args()

device_list = get_device_list()

''' Directory to save persistence diagrams '''
pkl_folder = f'./dgms/{args.net}/{args.net}_{args.dataset}'
pkl_folder += f'/{args.reduction}' if args.reduction is not None else ''
pkl_folder += f'/{args.metric}' if args.metric is not None else ''

# Build models
print('\n ==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device_list[0])
net.eval()

''' Prepare val data loader '''
test_transform = net._get_transform()
functloader = loader(f'{args.dataset}_test', batch_size=100, it=args.it, subset=args.subset, verbose=False, transform=test_transform) # subset size

''' Load checkpoint and get activations '''
assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'

with torch.no_grad():
    total_time = 0.

    epoch_iter = iter(vars(args)['chkpt_epochs'])
    for epoch in epoch_iter:
        if args.resume and (epoch <= args.resume_epoch):
            continue
        
        print(f'\n==> Loading checkpoint for epoch {epoch}...\n')
        
        checkpoint = torch.load(f'./checkpoint/{args.net}/{args.net}_{args.dataset}/ckpt_epoch_{epoch}.pt', map_location=device_list[0])
        
        # net.load_state_dict(checkpoint['net'])

        ''' Define passer and get activations '''
        # get activations and reduce dimensionality; compute distance adjacency matrix
        passer = Passer(net, functloader, None, device_list[0])
        activs = passer.get_function(reduction=args.reduction, device_list=device_list, corr=args.metric if not None else 'pearson', exp=args.exp, average=args.average)
        adj = adjacency(activs, metric=args.metric, device=device_list[0])

        if args.verbose:
            print(f'\n The dimension of the corrcoef matrix is {adj.size()[0], adj.size()[-1]} \n')
            print(f'Adj mean {adj.mean():.4f}, min {adj.min():.4f}, max {adj.max():.4f} \n')

        # convert to distance matrix for V-R filtration; metrics: (.5*(1 - adj))^.5 or (1 - |adj|)^.5
        adj = -adj
        adj += 1
        adj = torch.sqrt(.5 * adj.clamp(0., 1.))

        # convert to COO format for faster computation of persistence diagram
        indices = (adj<=1.).nonzero().numpy(force=True)
        i = list(zip(*indices))
        vals = adj[i].flatten().numpy(force=True)

        adj = coo_matrix((vals, i), shape=adj.shape) if len(i) > 0 else coo_matrix(([], ([], [])), shape=adj.shape)

        del indices, i, vals

        if args.verbose:
            print(f'\n The dimension of the COO distance matrix is {(len(adj.nonzero()[0]),)}\n')
            if adj.data.shape[0] != 0:
                print(f'adj mean {np.nanmean(adj.data):.4f}, min {np.nanmin(adj.data):.4f}, max {np.nanmax(adj.data):.4f}')
            else:
                print(f'adj empty! \n')

        # Compute persistence diagram
        comp_time = time.time()
        dgm = ripser_parallel(adj, metric="precomputed", maxdim=UPPER_DIM, n_threads=-1, collapse_edges=True)
        comp_time = time.time() - comp_time
        total_time += comp_time
        print(f'\n PH computation time: {comp_time/60:.2f} minutes \n')

        # free GPU memory
        del activs, adj, comp_time
        torch.cuda.empty_cache()

        dgm_gtda = _postprocess_diagrams([dgm["dgms"]], format="ripser", homology_dimensions=range(UPPER_DIM + 1), infinity_values=np.inf, reduced=True)[0]

        dgm_pkl_file = os.path.join(pkl_folder, f'dgm_epoch_{epoch}.pkl')
        if not os.path.exists(os.path.dirname(dgm_pkl_file)):
            os.makedirs(os.path.dirname(dgm_pkl_file))
        with open(dgm_pkl_file, 'wb') as f:
            pickle.dump(dgm_gtda, f, protocol=pickle.HIGHEST_PROTOCOL)

        del dgm, dgm_gtda

    print(f'\n Total computation time: {total_time/60:.2f} minutes \n')

    time_pkl_file = os.path.join(pkl_folder, f'time.pkl')
    if not os.path.exists(os.path.dirname(time_pkl_file)):
        os.makedirs(os.path.dirname(time_pkl_file))
    with open(time_pkl_file, 'ab') as f:
        pickle.dump(total_time, f, protocol=pickle.HIGHEST_PROTOCOL)

    del passer, net, functloader, total_time, checkpoint
