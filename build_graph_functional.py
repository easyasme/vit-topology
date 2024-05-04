from __future__ import print_function

import argparse
import time

from gph import ripser_parallel
# from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence
from gtda.homology._utils import _postprocess_diagrams
from gtda.utils import check_diagrams
from scipy.sparse import coo_matrix

from bettis import betti_nums
from config import UPPER_DIM
from graph import *
from loaders import *
from models.utils import get_model
from passers import Passer
from utils import *


parser = argparse.ArgumentParser(description='Build Graph and Compute Betti Numbers')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--subset', default=500, type=int, help='Subset size for building graph.')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: none, spearman, dcorr, or callable.')
parser.add_argument('--thresholds', default='0. 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--eps_thresh', default=1., type=float)
parser.add_argument('--reduction', default=None, type=str, help='Reductions: pca, umap or kmeans.')
parser.add_argument('--iter', default=0, type=int)
parser.add_argument('--verbose', default=0, type=int)

args = parser.parse_args()

device_list = []
if torch.cuda.device_count() > 1:
    device_list = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
    print("Using", torch.cuda.device_count(), "GPUs")
    for i, device in enumerate(device_list):
        print(f"Device {i}: {device}")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_list.append(device)
    print(f'Using {device}')

''' Directory to retrieve transformers '''
TRANS_DIR = f'./train_processing/{args.net}/{args.net}_{args.dataset}_ss{args.iter}' if args.dataset == 'imagenet' else f'./train_processing/{args.net}/{args.net}_{args.dataset}'
if not os.path.exists(TRANS_DIR):
    os.makedirs(TRANS_DIR)

''' Directory to save persistence diagrams '''
pkl_folder = f'./losses/{args.net}/{args.net}_{args.dataset}_ss{args.iter}' if args.dataset == 'imagenet' else f'./losses/{args.net}/{args.net}_{args.dataset}'
pkl_folder += f'/{args.reduction}' if args.reduction is not None else ''
pkl_folder += f'/{args.metric}' if args.metric is not None else ''

# Build models
print('\n ==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device_list[0])

''' Prepare criterion '''
criterion = nn.CrossEntropyLoss()

''' Prepare val data loader '''
trans_pkl_file = os.path.join(TRANS_DIR, f'train_transform.pkl')
with open(trans_pkl_file, 'rb') as f:
    train_transform = pickle.load(f)
functloader, _ = loader(f'{args.dataset}_test', batch_size=100, iter=args.iter, subset=args.subset, verbose=False, transform=train_transform) # subset size

''' Load checkpoint and get activations '''
assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
with torch.no_grad():
    total_time = 0.
    for epoch in vars(args)['chkpt_epochs']:
        print(f'\n==> Loading checkpoint for epoch {epoch}...\n')
        
        if args.dataset == 'imagenet':
            checkpoint = torch.load(f'./checkpoint/{args.net}/{args.net}_{args.dataset}_ss{args.iter}/ckpt_epoch_{epoch}.pt', map_location=device_list[0])
        else:
            checkpoint = torch.load(f'./checkpoint/{args.net}/{args.net}_{args.dataset}/ckpt_epoch_{epoch}.pt', map_location=device_list[0])
        
        net.load_state_dict(checkpoint['net'])
        net.requires_grad_(False)
        net.eval()

        ''' Define passer and get activations '''
        # get activations and reduce dimensionality; compute distance adjacency matrix
        passer = Passer(net, functloader, criterion, device_list[0])
        activs = passer.get_function(reduction=args.reduction, device_list=device_list) 
        adj = adjacency(activs, metric=args.metric, device=device_list[0])

        if args.verbose:
            print(f'\n The dimension of the corrcoef matrix is {adj.size()[0], adj.size()[-1]} \n')
            print(f'Adj mean {adj.mean():.4f}, min {adj.min():.4f}, max {adj.max():.4f} \n')

        # convert to distance matrix for V-R filtration
        adj = torch.abs(adj)
        adj *= -1
        adj += 1
        adj = torch.sqrt(adj.clamp(0, 1))

        # convert to COO format for faster computation of persistence diagram
        indices = (adj<1).nonzero().numpy(force=True)
        i = list(zip(*indices))
        vals = adj[i].flatten().numpy(force=True)

        adj = coo_matrix((vals, i), shape=adj.shape) if len(i) > 0 else coo_matrix(([], ([], [])), shape=adj.shape)

        del indices, i, vals

        if args.verbose:
            print(f'\n The dimension of the COO distance matrix is {adj.data.shape}\n')
            if adj.data.shape[0] != 0:
                print(f'adj mean {np.nanmean(adj.data):.4f}, min {np.nanmin(adj.data):.4f}, max {np.nanmax(adj.data):.4f}')
            else:
                print(f'adj empty! \n')

        # Compute persistence diagram
        comp_time = time.time()
        dgm = ripser_parallel(adj, metric="precomputed", maxdim=UPPER_DIM, n_threads=-1, collapse_edges=True)
        comp_time = time.time() - comp_time
        total_time += comp_time
        print(f'\n PH computation time: {comp_time/60:.5f} minutes \n')

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

    print(f'\n Total computation time: {total_time/60:.5f} minutes \n')

    time_pkl_file = os.path.join(pkl_folder, f'time.pkl')
    if not os.path.exists(os.path.dirname(time_pkl_file)):
        os.makedirs(os.path.dirname(time_pkl_file))
    with open(time_pkl_file, 'ab') as f:
        pickle.dump(total_time, f, protocol=pickle.HIGHEST_PROTOCOL)

    del passer, net, criterion, functloader, total_time, checkpoint
